#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Hypothesis: 6h Williams %R + 1d ADX regime filter + volume confirmation.
    # Williams %R identifies overbought/oversold conditions in trending markets.
    # Long: %R < -80 (oversold) + ADX > 25 (trending) + volume spike.
    # Short: %R > -20 (overbought) + ADX > 25 (trending) + volume spike.
    # Uses discrete size 0.25 to minimize fee churn. Target: 50-150 total trades over 4 years.
    # Works in bull/bear via ADX regime filter - only takes trades in strong trends.
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 6h data for Williams %R and volume (call ONCE before loop)
    df_6h = get_htf_data(prices, '6h')
    if len(df_6h) < 30:
        return np.zeros(n)
    
    # Get 1d data for ADX regime filter (call ONCE before loop)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    # Calculate 6h Williams %R(14)
    high_6h = df_6h['high'].values
    low_6h = df_6h['low'].values
    close_6h = df_6h['close'].values
    
    # Highest high and lowest low over 14 periods
    hh_14 = pd.Series(high_6h).rolling(window=14, min_periods=14).max().values
    ll_14 = pd.Series(low_6h).rolling(window=14, min_periods=14).min().values
    
    # Williams %R = (Highest High - Close) / (Highest High - Lowest Low) * -100
    williams_r = ((hh_14 - close_6h) / (hh_14 - ll_14)) * -100
    # Replace division by zero with -50 (neutral)
    williams_r = np.where((hh_14 - ll_14) == 0, -50, williams_r)
    
    # Calculate 6h volume mean (20-period) with min_periods
    volume_6h_series = pd.Series(df_6h['volume'].values)
    vol_ma_20_6h = volume_6h_series.rolling(window=20, min_periods=20).mean().values
    
    # Calculate 1d ADX(14) for regime filter
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # True Range
    tr1 = np.abs(high_1d[1:] - low_1d[:-1])
    tr2 = np.abs(high_1d[1:] - close_1d[:-1])
    tr3 = np.abs(low_1d[1:] - close_1d[:-1])
    tr = np.concatenate([[np.nan], np.maximum(tr1, np.maximum(tr2, tr3))])
    
    # Directional Movement
    dm_plus = np.where((high_1d[1:] - high_1d[:-1]) > (low_1d[:-1] - low_1d[1:]), 
                       np.maximum(high_1d[1:] - high_1d[:-1], 0), 0)
    dm_minus = np.where((low_1d[:-1] - low_1d[1:]) > (high_1d[1:] - high_1d[:-1]), 
                        np.maximum(low_1d[:-1] - low_1d[1:], 0), 0)
    dm_plus = np.concatenate([[np.nan], dm_plus])
    dm_minus = np.concatenate([[np.nan], dm_minus])
    
    # Smoothed TR, DM+, DM- (Wilder's smoothing = EMA with alpha=1/period)
    tr_series = pd.Series(tr)
    atr_14 = tr_series.ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    
    dm_plus_series = pd.Series(dm_plus)
    dm_plus_14 = dm_plus_series.ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    
    dm_minus_series = pd.Series(dm_minus)
    dm_minus_14 = dm_minus_series.ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    
    # Directional Indicators
    di_plus = 100 * dm_plus_14 / atr_14
    di_minus = 100 * dm_minus_14 / atr_14
    
    # DX and ADX
    dx = 100 * np.abs(di_plus - di_minus) / (di_plus + di_minus)
    dx_series = pd.Series(dx)
    adx_14 = dx_series.ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    
    # Align HTF indicators to 6h timeframe
    williams_r_aligned = align_htf_to_ltf(prices, df_6h, williams_r)
    vol_ma_aligned = align_htf_to_ltf(prices, df_6h, vol_ma_20_6h)
    adx_aligned = align_htf_to_ltf(prices, df_1d, adx_14)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(100, n):
        # Skip if data not ready
        if (np.isnan(williams_r_aligned[i]) or np.isnan(vol_ma_aligned[i]) or 
            np.isnan(adx_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Get current 6h volume for spike detection
        volume_6h_raw = df_6h['volume'].values
        vol_6h_aligned = align_htf_to_ltf(prices, df_6h, volume_6h_raw)
        
        # Volume filter: current 6h volume > 1.5 * 20-period mean (volume spike)
        volume_confirmation = vol_6h_aligned[i] > 1.5 * vol_ma_aligned[i]
        
        # Regime filter: ADX > 25 indicates trending market (good for mean reversion in trends)
        trending_regime = adx_aligned[i] > 25
        
        # Williams %R signals: mean reversion in trending markets
        williams_r_val = williams_r_aligned[i]
        
        # Oversold condition for long (%R < -80)
        oversold = williams_r_val < -80
        
        # Overbought condition for short (%R > -20)
        overbought = williams_r_val > -20
        
        # Entry conditions
        long_entry = oversold and volume_confirmation and trending_regime
        short_entry = overbought and volume_confirmation and trending_regime
        
        # Exit conditions: return to neutral territory (%R between -50 and -30)
        long_exit = williams_r_val > -50
        short_exit = williams_r_val < -30
        
        if long_entry and position != 1:
            position = 1
            signals[i] = 0.25
        elif short_entry and position != -1:
            position = -1
            signals[i] = -0.25
        elif position == 1 and long_exit:
            position = 0
            signals[i] = 0.0
        elif position == -1 and short_exit:
            position = 0
            signals[i] = 0.0
        else:
            # Hold current position
            if position == 1:
                signals[i] = 0.25
            elif position == -1:
                signals[i] = -0.25
            else:
                signals[i] = 0.0
    
    return signals

name = "6h_6h_1d_williams_r_adx_volume_v1"
timeframe = "6h"
leverage = 1.0
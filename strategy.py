#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h strategy combining 1-day ADX trend filter with 6h Williams %R mean reversion
# ADX > 25 on daily timeframe indicates strong trend (use for trend-following entries)
# ADX < 20 indicates ranging market (use for mean-reversion entries)
# Williams %R on 6h: > -20 overbought, < -80 oversold
# In trending markets (ADX>25): buy pullbacks to -80, sell rallies to -20
# In ranging markets (ADX<20): fade extremes at -80/-20 with confirmation
# Uses volume filter to avoid false signals
# Works in bull/bear: adapts to regime via ADX
# Target: 60-120 total trades over 4 years (15-30/year) with 0.25 position sizing

name = "6h_ADX_WilliamsR_Regime_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Calculate daily ADX ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    
    if len(df_1d) < 30:
        return np.zeros(n)
    
    # True Range
    tr1 = df_1d['high'] - df_1d['low']
    tr2 = abs(df_1d['high'] - df_1d['close'].shift(1))
    tr3 = abs(df_1d['low'] - df_1d['close'].shift(1))
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    
    # Directional Movement
    dm_plus = df_1d['high'] - df_1d['high'].shift(1)
    dm_minus = df_1d['low'].shift(1) - df_1d['low']
    dm_plus = dm_plus.where((dm_plus > dm_minus) & (dm_plus > 0), 0)
    dm_minus = dm_minus.where((dm_minus > dm_plus) & (dm_minus > 0), 0)
    
    # Smoothed values
    atr = tr.rolling(window=14, min_periods=14).mean()
    dm_plus_smooth = dm_plus.rolling(window=14, min_periods=14).sum()
    dm_minus_smooth = dm_minus.rolling(window=14, min_periods=14).sum()
    
    # Directional Indicators
    di_plus = 100 * dm_plus_smooth / atr
    di_minus = 100 * dm_minus_smooth / atr
    
    # DX and ADX
    dx = 100 * abs(di_plus - di_minus) / (di_plus + di_minus)
    adx = dx.rolling(window=14, min_periods=14).mean()
    
    adx_values = adx.values
    
    # Align daily ADX to 6h timeframe
    adx_aligned = align_htf_to_ltf(prices, df_1d, adx_values)
    
    # Williams %R on 6h (14-period)
    highest_high = pd.Series(high).rolling(window=14, min_periods=14).max()
    lowest_low = pd.Series(low).rolling(window=14, min_periods=14).min()
    williams_r = -100 * (highest_high - close) / (highest_high - lowest_low)
    williams_r = williams_r.fillna(-50).values  # neutral when undefined
    
    # Volume confirmation: >1.3x 20-period average
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean()
    volume_filter = volume > (1.3 * vol_ma_20)
    
    # Pre-compute session filter (08-20 UTC)
    hours = pd.DatetimeIndex(prices["open_time"]).hour
    session_filter = (hours >= 8) & (hours <= 20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(30, n):
        # Skip if any critical value is NaN or outside session
        if (np.isnan(adx_aligned[i]) or np.isnan(williams_r[i]) or 
            np.isnan(volume_filter[i]) or not session_filter[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        adx_val = adx_aligned[i]
        wr = williams_r[i]
        vol_ok = volume_filter[i]
        
        if position == 0:
            # Trending market (ADX > 25): mean reversion within trend
            if adx_val > 25 and vol_ok:
                # Long: pullback to oversold in uptrend
                if wr < -80 and wr > -85:  # entering oversold zone
                    signals[i] = 0.25
                    position = 1
                # Short: pullback to overbought in downtrend
                elif wr > -20 and wr < -15:  # entering overbought zone
                    signals[i] = -0.25
                    position = -1
            # Ranging market (ADX < 20): fade extremes
            elif adx_val < 20 and vol_ok:
                # Long: deep oversold bounce
                if wr < -85:
                    signals[i] = 0.25
                    position = 1
                # Short: deep overbought rejection
                elif wr > -15:
                    signals[i] = -0.25
                    position = -1
        elif position == 1:
            # Exit long: overbought or ADX weakening
            if wr > -25 or adx_val < 20:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: oversold or ADX weakening
            if wr < -75 or adx_val < 20:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals
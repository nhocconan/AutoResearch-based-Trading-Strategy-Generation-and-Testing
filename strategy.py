#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Williams %R reversal with 1d ADX trend filter and volume spike confirmation.
# Long when Williams %R crosses above -80 from oversold AND 1d ADX > 25 (trending) AND volume > 1.5x 20-bar average.
# Short when Williams %R crosses below -20 from overbought AND 1d ADX > 25 AND volume > 1.5x 20-bar average.
# Exit when Williams %R crosses -50 (mean reversion) or ADX < 20 (trend weakening).
# Uses discrete sizing 0.25 to minimize fee churn. Designed for 6h timeframe to capture medium-term reversals in both bull and bear markets.
# Williams %R identifies overbought/oversold conditions, ADX filters for trending environments to avoid whipsaws,
# Volume confirmation reduces false signals. This combination works in ranging markets (reversions at extremes) and trending markets (pullbacks in trend).

name = "6h_WilliamsR_ADX_VolumeReversal_v1"
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
    
    # Pre-compute session hours for efficiency
    hours = pd.DatetimeIndex(prices["open_time"]).hour
    
    # Load 1d data ONCE before loop for ADX trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 14:
        return np.zeros(n)
    
    # 1d ADX calculation (14-period)
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # True Range
    tr1 = high_1d - low_1d
    tr2 = np.abs(high_1d - np.roll(close_1d, 1))
    tr3 = np.abs(low_1d - np.roll(close_1d, 1))
    tr1[0] = 0  # first period has no previous close
    tr2[0] = 0
    tr3[0] = 0
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    
    # Directional Movement
    dm_plus = np.where((high_1d - np.roll(high_1d, 1)) > (np.roll(low_1d, 1) - low_1d), 
                       np.maximum(high_1d - np.roll(high_1d, 1), 0), 0)
    dm_minus = np.where((np.roll(low_1d, 1) - low_1d) > (high_1d - np.roll(high_1d, 1)), 
                        np.maximum(np.roll(low_1d, 1) - low_1d, 0), 0)
    dm_plus[0] = 0
    dm_minus[0] = 0
    
    # Smooth TR, DM+, DM- (Wilder's smoothing = EMA with alpha=1/period)
    atr = pd.Series(tr).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    dm_plus_smooth = pd.Series(dm_plus).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    dm_minus_smooth = pd.Series(dm_minus).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    
    # DI+ and DI-
    di_plus = 100 * dm_plus_smooth / (atr + 1e-10)
    di_minus = 100 * dm_minus_smooth / (atr + 1e-10)
    
    # DX and ADX
    dx = 100 * np.abs(di_plus - di_minus) / (di_plus + di_minus + 1e-10)
    adx = pd.Series(dx).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    adx_aligned = align_htf_to_ltf(prices, df_1d, adx)
    
    # Williams %R calculation (14-period) on 6h data
    highest_high = pd.Series(high).rolling(window=14, min_periods=14).max().values
    lowest_low = pd.Series(low).rolling(window=14, min_periods=14).min().values
    williams_r = -100 * (highest_high - close) / (highest_high - lowest_low + 1e-10)
    
    # Volume confirmation: current 6h volume > 1.5x 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 30  # warmup for Williams %R and ADX
    
    for i in range(start_idx, n):
        # Session filter: trade all sessions for 6h timeframe
        hour = hours[i]
        
        if np.isnan(williams_r[i]) or np.isnan(adx_aligned[i]) or np.isnan(vol_ma[i]):
            signals[i] = 0.0
            continue
        
        curr_close = close[i]
        curr_high = high[i]
        curr_low = low[i]
        curr_vol = volume[i]
        curr_vol_ma = vol_ma[i]
        
        if curr_vol_ma <= 0:
            signals[i] = 0.0
            continue
            
        volume_confirm = curr_vol > (curr_vol_ma * 1.5)
        
        # Williams %R signals
        wr = williams_r[i]
        wr_prev = williams_r[i-1] if i > 0 else wr
        
        # Cross above -80 (oversold reversal)
        cross_up_80 = (wr_prev <= -80) and (wr > -80)
        # Cross below -20 (overbought reversal)
        cross_down_20 = (wr_prev >= -20) and (wr < -20)
        # Cross above -50 (mean reversion exit)
        cross_up_50 = (wr_prev <= -50) and (wr > -50)
        # Cross below -50 (mean reversion exit)
        cross_down_50 = (wr_prev >= -50) and (wr < -50)
        
        # ADX trend conditions
        adx_val = adx_aligned[i]
        adx_trending = adx_val > 25
        adx_weakening = adx_val < 20
        
        # Entry conditions
        if position == 0:  # Flat - look for new entries
            # Long: Williams %R crosses above -80 FROM oversold AND ADX trending AND volume confirmation
            if (cross_up_80 and 
                adx_trending and 
                volume_confirm):
                signals[i] = 0.25
                position = 1
            # Short: Williams %R crosses below -20 FROM overbought AND ADX trending AND volume confirmation
            elif (cross_down_20 and 
                  adx_trending and 
                  volume_confirm):
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        
        elif position == 1:  # Long position
            # Exit: Williams %R crosses above -50 (overbought) OR ADX weakening
            if (cross_up_50 or 
                adx_weakening):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:  # Short position
            # Exit: Williams %R crosses below -50 (oversold) OR ADX weakening
            if (cross_down_50 or 
                adx_weakening):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals
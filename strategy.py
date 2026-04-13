#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h timeframe with 1-day Camarilla pivot levels, volume confirmation, and trend filter
# Strategy: Long when price breaks above R4 with volume > 1.5x average and ADX > 25
#           Short when price breaks below S4 with volume > 1.5x average and ADX > 25
#           Exit when price returns to pivot point (PP) or ADX drops below 20
#           Uses Camarilla levels for institutional breakout/breakdown levels
#           Volume surge confirms breakout strength, ADX filters for trending markets
#           Target: 50-150 total trades over 4 years (12-37/year) to balance opportunity and cost

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1d data for Camarilla pivot calculation
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate Camarilla pivot levels (using previous day's OHLC)
    # PP = (H + L + C) / 3
    # R4 = PP + (H - L) * 1.1/2
    # S4 = PP - (H - L) * 1.1/2
    pp_1d = (high_1d + low_1d + close_1d) / 3.0
    r4_1d = pp_1d + (high_1d - low_1d) * 1.1 / 2.0
    s4_1d = pp_1d - (high_1d - low_1d) * 1.1 / 2.0
    
    # Calculate ADX (14) on 1d for trend strength
    # +DM, -DM, TR
    high_diff = np.diff(high_1d, prepend=high_1d[0])
    low_diff = -np.diff(low_1d, prepend=low_1d[0])
    plus_dm = np.where((high_diff > low_diff) & (high_diff > 0), high_diff, 0)
    minus_dm = np.where((low_diff > high_diff) & (low_diff > 0), low_diff, 0)
    
    tr1 = np.abs(np.diff(high_1d, prepend=high_1d[0]))
    tr2 = np.abs(np.diff(low_1d, prepend=low_1d[0]))
    tr3 = np.abs(high_1d[1:] - low_1d[:-1])
    tr3 = np.concatenate([[tr3[0]] if len(tr3) > 0 else [0], tr3])
    tr = np.maximum(np.maximum(tr1, tr2), tr3)
    
    # Wilder's smoothing (alpha = 1/period)
    def wilders_smoothing(data, period):
        alpha = 1.0 / period
        result = np.full_like(data, np.nan)
        if len(data) >= period:
            result[period-1] = np.nanmean(data[:period])
            for i in range(period, len(data)):
                result[i] = alpha * data[i] + (1 - alpha) * result[i-1]
        return result
    
    atr_1d = wilders_smoothing(tr, 14)
    plus_di_1d = 100 * wilders_smoothing(plus_dm, 14) / (atr_1d + 1e-10)
    minus_di_1d = 100 * wilders_smoothing(minus_dm, 14) / (atr_1d + 1e-10)
    dx_1d = 100 * np.abs(plus_di_1d - minus_di_1d) / (plus_di_1d + minus_di_1d + 1e-10)
    adx_1d = wilders_smoothing(dx_1d, 14)
    
    # Volume average (20-period)
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    # Align 1d indicators to 6h timeframe
    r4_1d_aligned = align_htf_to_ltf(prices, df_1d, r4_1d)
    s4_1d_aligned = align_htf_to_ltf(prices, df_1d, s4_1d)
    pp_1d_aligned = align_htf_to_ltf(prices, df_1d, pp_1d)
    adx_1d_aligned = align_htf_to_ltf(prices, df_1d, adx_1d)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    position_size = 0.25  # 25% of capital
    
    for i in range(50, n):
        # Skip if data not ready
        if (np.isnan(r4_1d_aligned[i]) or 
            np.isnan(s4_1d_aligned[i]) or 
            np.isnan(pp_1d_aligned[i]) or 
            np.isnan(adx_1d_aligned[i]) or 
            np.isnan(vol_ma_20[i])):
            signals[i] = 0.0
            continue
        
        # Volume surge condition
        volume_surge = volume[i] > 1.5 * vol_ma_20[i]
        
        # Breakout conditions
        long_breakout = close[i] > r4_1d_aligned[i]
        short_breakout = close[i] < s4_1d_aligned[i]
        
        # Trend filter
        strong_trend = adx_1d_aligned[i] > 25
        
        # Exit conditions: return to pivot or loss of trend
        exit_long = position == 1 and (close[i] <= pp_1d_aligned[i] or adx_1d_aligned[i] < 20)
        exit_short = position == -1 and (close[i] >= pp_1d_aligned[i] or adx_1d_aligned[i] < 20)
        
        # Entry logic
        long_entry = long_breakout and volume_surge and strong_trend
        short_entry = short_breakout and volume_surge and strong_trend
        
        # Execute signals
        if long_entry and position != 1:
            position = 1
            signals[i] = position_size
        elif short_entry and position != -1:
            position = -1
            signals[i] = -position_size
        elif exit_long or exit_short:
            position = 0
            signals[i] = 0.0
        # Hold current position
        else:
            if position == 1:
                signals[i] = position_size
            elif position == -1:
                signals[i] = -position_size
            else:
                signals[i] = 0.0
    
    return signals

name = "6h_1d_camarilla_breakout_volume_adx_v1"
timeframe = "6h"
leverage = 1.0
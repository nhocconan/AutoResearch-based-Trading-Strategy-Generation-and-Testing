#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1d Donchian(20) breakout with 1w volume confirmation and ADX regime filter
# In strong trends (ADX > 25): breakout above/below Donchian H/L with volume confirmation
# In weak trends/ranges (ADX <= 25): no new entries, hold or exit
# Uses discrete position sizing 0.30 to limit trades and reduce fee drag
# Works in bull/bear markets: breakout catches strong moves, ADX filter avoids chop

name = "1d_1w_donchian_breakout_volume_adx_v1"
timeframe = "1d"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Load 1w data ONCE before loop
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 30:
        return np.zeros(n)
    
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    volume_1w = df_1w['volume'].values
    
    # Calculate 1w ADX(14)
    # True Range
    tr1 = np.abs(high_1w[1:] - low_1w[:-1])
    tr2 = np.abs(high_1w[1:] - close_1w[:-1])
    tr3 = np.abs(low_1w[1:] - close_1w[:-1])
    tr = np.concatenate([[np.nan], np.maximum(tr1, np.maximum(tr2, tr3))])
    
    # Directional Movement
    up_move = high_1w[1:] - high_1w[:-1]
    down_move = low_1w[:-1] - low_1w[1:]
    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0.0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0.0)
    plus_dm = np.concatenate([[0.0], plus_dm])
    minus_dm = np.concatenate([[0.0], minus_dm])
    
    # Wilder's smoothing
    def wilders_smoothing(values, period):
        if len(values) < period:
            return np.full(len(values), np.nan)
        alpha = 1.0 / period
        result = np.full(len(values), np.nan)
        result[period-1] = np.nanmean(values[:period])
        for i in range(period, len(values)):
            result[i] = alpha * values[i] + (1 - alpha) * result[i-1]
        return result
    
    atr_1w = wilders_smoothing(tr, 14)
    plus_di_1w = 100 * wilders_smoothing(plus_dm, 14) / atr_1w
    minus_di_1w = 100 * wilders_smoothing(minus_dm, 14) / atr_1w
    dx_1w = 100 * np.abs(plus_di_1w - minus_di_1w) / (plus_di_1w + minus_di_1w)
    adx_1w = wilders_smoothing(dx_1w, 14)
    
    # Calculate 1w average volume (20-period)
    volume_s_1w = pd.Series(volume_1w)
    avg_volume_1w = volume_s_1w.rolling(window=20, min_periods=20).mean().values
    
    # Calculate 1d Donchian channels (20-period)
    highest_high = pd.Series(high).rolling(window=20, min_periods=20).max().values
    lowest_low = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # Align 1w indicators to 1d timeframe
    adx_1w_aligned = align_htf_to_ltf(prices, df_1w, adx_1w)
    avg_volume_1w_aligned = align_htf_to_ltf(prices, df_1w, avg_volume_1w)
    
    # Volume confirmation: current volume > 1.5x average weekly volume
    volume_confirmed = volume > 1.5 * avg_volume_1w_aligned
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(100, n):
        # Skip if any required data is invalid
        if (np.isnan(adx_1w_aligned[i]) or np.isnan(highest_high[i]) or 
            np.isnan(lowest_low[i]) or np.isnan(volume_confirmed[i])):
            signals[i] = 0.0
            continue
        
        if position == 1:  # Long position
            # Exit long if price breaks below Donchian low or ADX weakens
            if close[i] < lowest_low[i] or adx_1w_aligned[i] <= 25:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.30
        elif position == -1:  # Short position
            # Exit short if price breaks above Donchian high or ADX weakens
            if close[i] > highest_high[i] or adx_1w_aligned[i] <= 25:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.30
        else:  # Flat
            # Enter long on breakout above Donchian high with volume confirmation and strong trend
            if close[i] > highest_high[i] and volume_confirmed[i] and adx_1w_aligned[i] > 25:
                position = 1
                signals[i] = 0.30
            # Enter short on breakout below Donchian low with volume confirmation and strong trend
            elif close[i] < lowest_low[i] and volume_confirmed[i] and adx_1w_aligned[i] > 25:
                position = -1
                signals[i] = -0.30
    
    return signals
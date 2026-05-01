#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1d Donchian(20) breakout with 1w volume confirmation and ATR trailing stop.
# Long when price breaks above 20-day Donchian high with volume > 1.5x weekly average volume.
# Short when price breaks below 20-day Donchian low with volume confirmation.
# Uses discrete sizing 0.25. ATR(14) trailing stop: signal→0 when price retraces 2.5*ATR from extreme.
# Donchian channels calculated from prior completed 1d bar. Weekly volume filter ensures institutional participation.
# Works in bull (breakouts with volume) and bear (breakdowns with volume) regimes.
# Target: 30-100 trades over 4 years (7-25/year) on 1d timeframe.

name = "1d_Donchian20_Breakout_1wVolume_ATRStop_v1"
timeframe = "1d"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Calculate ATR(14) for trailing stop
    tr1 = high[1:] - low[1:]
    tr2 = np.abs(high[1:] - close[:-1])
    tr3 = np.abs(low[1:] - close[:-1])
    tr_first = np.max([high[0] - low[0], np.abs(high[0] - close[0]), np.abs(low[0] - close[0])])
    tr = np.concatenate([[tr_first], np.maximum(tr1, np.maximum(tr2, tr3))])
    atr = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    # Load 1d data ONCE before loop for Donchian channels
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate Donchian(20) channels using previous completed 1d bar
    # Upper = max(high of last 20 days), Lower = min(low of last 20 days)
    # Shift by 1 to use only completed bars (no look-ahead)
    upper_20 = pd.Series(high_1d).rolling(window=20, min_periods=20).max().shift(1).values
    lower_20 = pd.Series(low_1d).rolling(window=20, min_periods=20).min().shift(1).values
    
    # Load 1w data ONCE before loop for volume confirmation (HTF)
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 1:
        return np.zeros(n)
    
    # Calculate weekly volume average
    vol_1w = df_1w['volume'].values
    vol_ma_1w = pd.Series(vol_1w).rolling(window=4, min_periods=4).mean().values  # ~4 weeks per month
    
    # Align 1w volume MA to 1d timeframe
    vol_ma_1w_aligned = align_htf_to_ltf(prices, df_1w, vol_ma_1w)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    long_extreme = 0.0   # track highest high for long trailing stop
    short_extreme = 0.0  # track lowest low for short trailing stop
    
    # Start after warmup for Donchian and ATR
    start_idx = 30
    
    for i in range(start_idx, n):
        if (np.isnan(atr[i]) or 
            np.isnan(upper_20[i]) or 
            np.isnan(lower_20[i]) or 
            np.isnan(vol_ma_1w_aligned[i])):
            signals[i] = 0.0
            continue
        
        curr_close = close[i]
        curr_high = high[i]
        curr_low = low[i]
        curr_volume = volume[i]
        
        # Volume confirmation: current volume > 1.5x weekly average volume
        if vol_ma_1w_aligned[i] <= 0 or np.isnan(vol_ma_1w_aligned[i]):
            volume_confirm = False
        else:
            volume_confirm = curr_volume > (vol_ma_1w_aligned[i] * 1.5)
        
        upper_level = upper_20[i]
        lower_level = lower_20[i]
        
        if position == 0:  # Flat - look for new entries
            # Long: Donchian breakout up AND volume confirmation
            if (curr_high > upper_level and volume_confirm):
                signals[i] = 0.25
                position = 1
                entry_price = curr_close
                long_extreme = curr_high
            # Short: Donchian breakout down AND volume confirmation
            elif (curr_low < lower_level and volume_confirm):
                signals[i] = -0.25
                position = -1
                entry_price = curr_close
                short_extreme = curr_low
            else:
                signals[i] = 0.0
        
        elif position == 1:  # Long position
            # Update extreme for trailing stop
            if curr_high > long_extreme:
                long_extreme = curr_high
            
            # ATR trailing stop: signal→0 when price retraces 2.5*ATR from extreme
            if curr_close < long_extreme - 2.5 * atr[i]:
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
                long_extreme = 0.0
            # Exit: price re-enters Donchian channel (mean reversion signal)
            elif curr_low >= lower_level and curr_low <= upper_level:
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
                long_extreme = 0.0
            else:
                signals[i] = 0.25
        
        elif position == -1:  # Short position
            # Update extreme for trailing stop
            if curr_low < short_extreme:
                short_extreme = curr_low
            
            # ATR trailing stop: signal→0 when price retraces 2.5*ATR from extreme
            if curr_close > short_extreme + 2.5 * atr[i]:
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
                short_extreme = 0.0
            # Exit: price re-enters Donchian channel (mean reversion signal)
            elif curr_high >= lower_level and curr_high <= upper_level:
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
                short_extreme = 0.0
            else:
                signals[i] = -0.25
    
    return signals
#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Donchian(20) breakout with 1d EMA50 trend filter and volume spike confirmation.
# Long when price breaks above upper Donchian + 1d EMA50 uptrend + volume > 2.0x 20-bar average.
# Short when price breaks below lower Donchian + 1d EMA50 downtrend + volume > 2.0x 20-bar average.
# ATR trailing stop (2.0x) for risk management.
# Targets 75-200 total trades over 4 years (19-50/year) with discrete position sizing (0.25).
# Uses 1d HTF for trend filter (more stable than 4h) and stricter volume confirmation (2.0x) to reduce overtrading.
# Donchian channels provide clear structure; breakouts with volume confirm conviction.

name = "4h_Donchian20_1dEMA50_Trend_VolumeConfirm_v1"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Load 1d data ONCE before loop for EMA50 trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # Calculate 1d EMA50 for trend filter
    close_1d = df_1d['close'].values
    ema_50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # Calculate Donchian(20) from previous 20 periods (avoid look-ahead)
    # Upper = max(high[-20:-1]), Lower = min(low[-20:-1])
    high_ma_20 = pd.Series(high).rolling(window=20, min_periods=1).max().values
    low_ma_20 = pd.Series(low).rolling(window=20, min_periods=1).min().values
    # Shift by 1 to use only past data
    upper_channel = np.concatenate([[high[0]], high_ma_20[:-1]])
    lower_channel = np.concatenate([[low[0]], low_ma_20[:-1]])
    
    # Volume confirmation: volume > 2.0x 20-period average (stricter to reduce overtrading)
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=1).mean().values
    volume_confirm = volume > (2.0 * vol_ma_20)
    
    # ATR for trailing stop
    tr1 = high[1:] - low[1:]
    tr2 = np.abs(high[1:] - close[:-1])
    tr3 = np.abs(low[1:] - close[:-1])
    tr = np.concatenate([[np.max([tr1[0], tr2[0], tr3[0]])], np.maximum(tr1, np.maximum(tr2, tr3))])
    atr = pd.Series(tr).ewm(span=14, adjust=False, min_periods=14).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    highest_since_entry = 0.0
    lowest_since_entry = 0.0
    
    start_idx = 50  # warmup for EMA50 and indicators
    
    for i in range(start_idx, n):
        # Skip if indicators not available
        if np.isnan(ema_50_aligned[i]) or np.isnan(upper_channel[i]) or np.isnan(lower_channel[i]):
            if position == 1:
                signals[i] = 0.25
            elif position == -1:
                signals[i] = -0.25
            continue
        
        # Regime filter: price above/below 1d EMA50 determines trend direction
        is_uptrend = close[i] > ema_50_aligned[i]
        is_downtrend = close[i] < ema_50_aligned[i]
        
        curr_close = close[i]
        curr_high = high[i]
        curr_low = low[i]
        curr_atr = atr[i]
        curr_volume_confirm = volume_confirm[i]
        
        if position == 0:  # Flat - look for new entries
            # Long: price breaks above upper Donchian + uptrend + volume confirmation
            if curr_high > upper_channel[i] and is_uptrend and curr_volume_confirm:
                signals[i] = 0.25
                position = 1
                highest_since_entry = curr_high
            # Short: price breaks below lower Donchian + downtrend + volume confirmation
            elif curr_low < lower_channel[i] and is_downtrend and curr_volume_confirm:
                signals[i] = -0.25
                position = -1
                lowest_since_entry = curr_low
        
        elif position == 1:  # Long position
            # Update highest high since entry
            if curr_high > highest_since_entry:
                highest_since_entry = curr_high
            
            # Trailing stop: 2.0 * ATR below highest since entry
            if curr_close < highest_since_entry - 2.0 * curr_atr:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:  # Short position
            # Update lowest low since entry
            if curr_low < lowest_since_entry:
                lowest_since_entry = curr_low
            
            # Trailing stop: 2.0 * ATR above lowest since entry
            if curr_close > lowest_since_entry + 2.0 * curr_atr:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals
#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1d Donchian(20) breakout with 1w EMA34 trend filter and volume confirmation
# Uses 1d timeframe for signal generation and 1w for trend filter (HTF)
# Volume confirmation (1.5x 20-period average on 1d) ensures institutional participation
# Designed for low trade frequency (target: 50-100 total trades over 4 years) to minimize fee drag
# Works in bull markets via trend-aligned breakouts and in bear markets via strict regime filter

name = "1d_Donchian20_1wEMA34_Volume_Confirm_v1"
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
    
    # Load 1w data ONCE before loop for EMA trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 34:
        return np.zeros(n)
    
    # Calculate 1w EMA34 for trend filter
    close_1w = df_1w['close'].values
    ema_34_1w = pd.Series(close_1w).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_34_1w)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after warmup (need enough for Donchian and volume MA)
    start_idx = 50
    
    for i in range(start_idx, n):
        # Calculate Donchian channels from last 20 periods (excluding current)
        lookback_start = max(0, i - 20)
        lookback_end = i  # exclude current bar
        
        if lookback_end - lookback_start < 20:
            signals[i] = 0.0
            continue
            
        highest_high = np.max(high[lookback_start:lookback_end])
        lowest_low = np.min(low[lookback_start:lookback_end])
        
        # Volume confirmation: 1.5x 20-period average
        if i >= 20:
            vol_ma = np.mean(volume[i-20:i])
            volume_confirm = volume[i] > (vol_ma * 1.5)
        else:
            volume_confirm = False
        
        if position == 0:  # Flat - look for new entries
            # Long: Price breaks above 20-period high + price > 1w EMA34 + volume confirm
            if close[i] > highest_high and close[i] > ema_34_1w_aligned[i] and volume_confirm:
                signals[i] = 0.25
                position = 1
            # Short: Price breaks below 20-period low + price < 1w EMA34 + volume confirm
            elif close[i] < lowest_low and close[i] < ema_34_1w_aligned[i] and volume_confirm:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        
        elif position == 1:  # Long position
            # Exit: Price breaks below 20-period low (trailing stop)
            if close[i] < lowest_low:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:  # Short position
            # Exit: Price breaks above 20-period high (trailing stop)
            if close[i] > highest_high:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals
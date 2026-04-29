#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1d Donchian(20) breakout + 1w EMA50 trend filter + volume confirmation
# Long when price breaks above Donchian(20) high AND price > 1w EMA50 AND volume > 2.0x 20-period average
# Short when price breaks below Donchian(20) low AND price < 1w EMA50 AND volume > 2.0x 20-period average
# Uses discrete position sizing (0.25) to minimize fee drag and ATR-based trailing stop (2.0x ATR)
# Works in both bull and bear markets by combining breakout momentum with trend filter
# Target: 1d timeframe, HTF = 1w, expected trades: 30-100 over 4 years (7-25/year)

name = "1d_Donchian20_Breakout_1wEMA50_VolumeConfirm_v1"
timeframe = "1d"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Load HTF data ONCE before loop
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 60:
        return np.zeros(n)
    
    # Calculate 1w EMA50 for trend filter
    close_1w = df_1w['close'].values
    ema_50_1w = pd.Series(close_1w).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_50_1w)
    
    # Calculate ATR for stoploss (using 14-period)
    tr1 = high[1:] - low[1:]
    tr2 = np.abs(high[1:] - close[:-1])
    tr3 = np.abs(low[1:] - close[:-1])
    tr_first = np.max([high[0] - low[0], np.abs(high[0] - close[0]), np.abs(low[0] - close[0])])
    tr = np.concatenate([[tr_first], np.maximum(tr1, np.maximum(tr2, tr3))])
    atr = pd.Series(tr).ewm(span=14, adjust=False, min_periods=14).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    highest_high_since_entry = 0.0
    lowest_low_since_entry = 0.0
    
    start_idx = max(100, 20, 50, 14)  # warmup for indicators
    
    for i in range(start_idx, n):
        curr_close = close[i]
        curr_high = high[i]
        curr_low = low[i]
        curr_ema_1w = ema_50_1w_aligned[i]
        curr_atr = atr[i]
        
        # Volume confirmation: current volume > 2.0x 20-period average
        if i >= 20:
            vol_ma_20 = np.mean(volume[i-20:i])
        else:
            vol_ma_20 = 0.0
        vol_spike = volume[i] > 2.0 * vol_ma_20 if vol_ma_20 > 0 else False
        
        # Donchian(20) channels - calculate using lookback window
        if i >= 20:
            donchian_high = np.max(high[i-20:i])
            donchian_low = np.min(low[i-20:i])
        else:
            donchian_high = curr_high
            donchian_low = curr_low
        
        # Handle exits and stoploss
        if position == 1:  # Long position
            # Update highest high since entry
            highest_high_since_entry = max(highest_high_since_entry, curr_high)
            # Trailing stop: 2.0 * ATR below highest high
            stop_price = highest_high_since_entry - 2.0 * curr_atr
            # Exit conditions: price below trailing stop OR price breaks below Donchian low
            if curr_close < stop_price or curr_close < donchian_low:
                signals[i] = 0.0
                position = 0
                highest_high_since_entry = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Update lowest low since entry
            lowest_low_since_entry = min(lowest_low_since_entry, curr_low)
            # Trailing stop: 2.0 * ATR above lowest low
            stop_price = lowest_low_since_entry + 2.0 * curr_atr
            # Exit conditions: price above trailing stop OR price breaks above Donchian high
            if curr_close > stop_price or curr_close > donchian_high:
                signals[i] = 0.0
                position = 0
                lowest_low_since_entry = 0.0
            else:
                signals[i] = -0.25
                
        else:  # Flat - look for new entries
            # Long entry: price breaks above Donchian high AND price > 1w EMA50 AND volume spike
            if (curr_close > donchian_high and 
                curr_close > curr_ema_1w and 
                vol_spike):
                signals[i] = 0.25
                position = 1
                highest_high_since_entry = curr_high
            # Short entry: price breaks below Donchian low AND price < 1w EMA50 AND volume spike
            elif (curr_close < donchian_low and 
                  curr_close < curr_ema_1w and 
                  vol_spike):
                signals[i] = -0.25
                position = -1
                lowest_low_since_entry = curr_low
            else:
                signals[i] = 0.0
    
    return signals
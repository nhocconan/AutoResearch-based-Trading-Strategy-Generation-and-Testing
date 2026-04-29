#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h Donchian(20) breakout with 1d EMA50 trend filter and volume confirmation
# Uses daily EMA50 for trend to avoid counter-trend trades in ranging/bear markets
# Donchian breakout from 20-period (10-day) high/low captures momentum
# Volume spike (>1.5x 20-period average) confirms breakout validity
# ATR-based trailing stop (2.5x ATR) manages risk and locks in profits
# Designed for low trade frequency (target: 50-150 total over 4 years) to minimize fee drag
# Works in bull markets via trend-following breaks and in bear markets via trend filter avoidance

name = "12h_Donchian_Breakout_1dEMA50_VolumeConfirm_ATRStop_v1"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Load HTF data ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # Calculate 1d EMA50 for trend filter
    close_1d = df_1d['close'].values
    ema_50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # Calculate ATR for stoploss (using 14-period)
    tr1 = high[1:] - low[1:]
    tr2 = np.abs(high[1:] - close[:-1])
    tr3 = np.abs(low[1:] - close[:-1])
    tr_first = np.max([high[0] - low[0], np.abs(high[0] - close[0]), np.abs(low[0] - close[0])])
    tr = np.concatenate([[tr_first], np.maximum(tr1, np.maximum(tr2, tr3))])
    atr = pd.Series(tr).ewm(span=14, adjust=False, min_periods=14).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    highest_since_entry = 0.0
    lowest_since_entry = 0.0
    
    start_idx = 50  # warmup for EMA and Donchian
    
    for i in range(start_idx, n):
        # Calculate Donchian channels from last 20 periods
        lookback = 20
        if i < lookback:
            signals[i] = 0.0
            continue
            
        donchian_high = np.max(high[i-lookback:i])
        donchian_low = np.min(low[i-lookback:i])
        
        curr_close = close[i]
        curr_high = high[i]
        curr_low = low[i]
        curr_ema_1d = ema_50_1d_aligned[i]
        curr_atr = atr[i]
        
        # Volume confirmation: current volume > 1.5x 20-period average
        vol_lookback = 20
        if i >= vol_lookback:
            vol_ma_20 = np.mean(volume[i-vol_lookback:i])
        else:
            vol_ma_20 = 0.0
        vol_spike = volume[i] > 1.5 * vol_ma_20 if vol_ma_20 > 0 else False
        
        # Handle exits and trailing stop
        if position == 1:  # Long position
            # Update highest high since entry
            highest_since_entry = max(highest_since_entry, curr_high)
            # Trailing stop: 2.5 * ATR below highest high
            stop_price = highest_since_entry - 2.5 * curr_atr
            # Exit conditions: price below Donchian low OR stoploss hit
            if curr_close < donchian_low or curr_close < stop_price:
                signals[i] = 0.0
                position = 0
                highest_since_entry = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Update lowest low since entry
            lowest_since_entry = min(lowest_since_entry, curr_low)
            # Trailing stop: 2.5 * ATR above lowest low
            stop_price = lowest_since_entry + 2.5 * curr_atr
            # Exit conditions: price above Donchian high OR stoploss hit
            if curr_close > donchian_high or curr_close > stop_price:
                signals[i] = 0.0
                position = 0
                lowest_since_entry = 0.0
            else:
                signals[i] = -0.25
                
        else:  # Flat - look for new entries
            # Long entry: price breaks above Donchian high AND price > 1d EMA50 AND volume spike
            if curr_close > donchian_high and curr_close > curr_ema_1d and vol_spike:
                signals[i] = 0.25
                position = 1
                entry_price = curr_close
                highest_since_entry = curr_high
            # Short entry: price breaks below Donchian low AND price < 1d EMA50 AND volume spike
            elif curr_close < donchian_low and curr_close < curr_ema_1d and vol_spike:
                signals[i] = -0.25
                position = -1
                entry_price = curr_close
                lowest_since_entry = curr_low
            else:
                signals[i] = 0.0
    
    return signals
#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1d Donchian(20) breakout + 1w EMA50 trend + volume confirmation + ATR(14) trailing stop
# Donchian channels provide robust structure; weekly EMA50 filters for higher timeframe trend;
# volume confirms breakout strength; ATR-based trailing stop manages risk in both bull and bear markets.
# Target: 20-30 trades/year (80-120 total over 4 years) to balance opportunity and fee drag.

name = "1d_Donchian20_Breakout_1wEMA50_VolumeSpike_ATRTrail_v1"
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
    open_price = prices['open'].values
    
    # Load HTF data ONCE before loop for 1w calculations
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 50:
        return np.zeros(n)
    
    # Calculate 1w EMA50 for trend filter
    ema_50_1w = pd.Series(df_1w['close'].values).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_50_1w)
    
    # Calculate ATR(14) for trailing stop
    tr1 = high[1:] - low[1:]
    tr2 = np.abs(high[1:] - close[:-1])
    tr3 = np.abs(low[1:] - close[:-1])
    tr = np.concatenate([[np.nan], np.maximum(tr1, np.maximum(tr2, tr3))])
    atr = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    # Calculate Donchian channels (20-period)
    highest_high_20 = pd.Series(high).rolling(window=20, min_periods=20).max().values
    lowest_low_20 = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # Volume confirmation: volume > 1.5x 20-period average
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_confirm = volume > (1.5 * vol_ma_20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    atr_at_entry = 0.0
    max_high_since_entry = 0.0  # For trailing stop
    min_low_since_entry = 0.0   # For trailing stop
    
    start_idx = max(50, 20, 14)  # warmup for EMA50, Donchian, ATR
    
    for i in range(start_idx, n):
        # Skip if HTF data not available
        if np.isnan(ema_50_1w_aligned[i]) or np.isnan(highest_high_20[i]) or np.isnan(lowest_low_20[i]):
            signals[i] = 0.0
            continue
            
        curr_close = close[i]
        curr_high = high[i]
        curr_low = low[i]
        curr_open = open_price[i]
        curr_ema_50_1w = ema_50_1w_aligned[i]
        curr_atr = atr[i]
        curr_donchian_high = highest_high_20[i]
        curr_donchian_low = lowest_low_20[i]
        curr_volume_confirm = volume_confirm[i]
        
        # Handle position exits and stops
        if position == 1:  # Long position
            # Update trailing stop: highest high since entry
            max_high_since_entry = max(max_high_since_entry, curr_high)
            # Trailing stop: ATR-based
            trail_stop = max_high_since_entry - 2.5 * curr_atr
            
            # Exit conditions:
            # 1. Stoploss hit (trailing)
            # 2. Price crosses below 1w EMA50 (trend change)
            # 3. Price drops below Donchian low (breakout failed)
            if (curr_low <= trail_stop or
                curr_close < curr_ema_50_1w or
                curr_close < curr_donchian_low):
                signals[i] = 0.0
                position = 0
                max_high_since_entry = 0.0
                min_low_since_entry = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Update trailing stop: lowest low since entry
            min_low_since_entry = min(min_low_since_entry, curr_low)
            # Trailing stop: ATR-based
            trail_stop = min_low_since_entry + 2.5 * curr_atr
            
            # Exit conditions:
            # 1. Stoploss hit (trailing)
            # 2. Price crosses above 1w EMA50 (trend change)
            # 3. Price rises above Donchian high (breakout failed)
            if (curr_high >= trail_stop or
                curr_close > curr_ema_50_1w or
                curr_close > curr_donchian_high):
                signals[i] = 0.0
                position = 0
                max_high_since_entry = 0.0
                min_low_since_entry = 0.0
            else:
                signals[i] = -0.25
                
        else:  # Flat - look for new entries
            # Long entry: price breaks above Donchian high + above 1w EMA50 + volume confirm
            if (curr_close > curr_donchian_high and
                curr_close > curr_ema_50_1w and
                curr_volume_confirm):
                signals[i] = 0.25
                position = 1
                entry_price = curr_close
                atr_at_entry = curr_atr
                max_high_since_entry = curr_high
                min_low_since_entry = curr_low
            # Short entry: price breaks below Donchian low + below 1w EMA50 + volume confirm
            elif (curr_close < curr_donchian_low and
                  curr_close < curr_ema_50_1w and
                  curr_volume_confirm):
                signals[i] = -0.25
                position = -1
                entry_price = curr_close
                atr_at_entry = curr_atr
                max_high_since_entry = curr_high
                min_low_since_entry = curr_low
            else:
                signals[i] = 0.0
    
    return signals
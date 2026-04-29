#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Donchian(20) breakout + 12h EMA50 trend + volume spike + ATR stoploss
# Long when: price breaks above Donchian(20) high AND price > 12h EMA50 AND volume > 2.0x avg
# Short when: price breaks below Donchian(20) low AND price < 12h EMA50 AND volume > 2.0x avg
# Exit on ATR(14) trailing stop or opposite breakout
# Uses discrete sizing (0.25) to minimize fee churn. Works in bull/bear via 12h trend filter.
# Timeframe: 4h (primary), HTF: 12h for EMA50 trend.

name = "4h_Donchian20_Breakout_12hEMA50_VolumeSpike_ATRStop_v1"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Load HTF data ONCE before loop for 12h EMA50 trend
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 50:
        return np.zeros(n)
    
    # Calculate 12h EMA50
    ema_50_12h = pd.Series(df_12h['close'].values).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_12h_aligned = align_htf_to_ltf(prices, df_12h, ema_50_12h)
    
    # Donchian channels (20-period)
    highest_high = pd.Series(high).rolling(window=20, min_periods=20).max().values
    lowest_low = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # Volume confirmation: volume > 2.0x 20-period average
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_confirm = volume > (2.0 * vol_ma_20)
    
    # ATR(14) for stoploss
    tr1 = np.abs(high - low)
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = 0
    atr = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    highest_since_entry = 0.0
    lowest_since_entry = 0.0
    
    start_idx = max(20, 50, 20, 14)  # warmup for Donchian, EMA50, volume MA, ATR
    
    for i in range(start_idx, n):
        # Skip if HTF data not available
        if np.isnan(ema_50_12h_aligned[i]) or np.isnan(atr[i]):
            signals[i] = 0.0
            continue
            
        curr_close = close[i]
        curr_high = high[i]
        curr_low = low[i]
        curr_highest = highest_high[i]
        curr_lowest = lowest_low[i]
        curr_ema_50_12h = ema_50_12h_aligned[i]
        curr_volume_confirm = volume_confirm[i]
        curr_atr = atr[i]
        
        # Handle position exits
        if position == 1:  # Long position
            # Update highest since entry
            if curr_close > highest_since_entry:
                highest_since_entry = curr_close
            # Exit conditions:
            # 1. ATR trailing stop: price < highest_since_entry - 2.5 * ATR
            # 2. Opposite Donchian breakout: price < lowest_low (20-period low)
            # 3. Trend filter fails: price < 12h EMA50
            if (curr_close < highest_since_entry - 2.5 * curr_atr or
                curr_close < curr_lowest or
                curr_close < curr_ema_50_12h):
                signals[i] = 0.0
                position = 0
                highest_since_entry = 0.0
                lowest_since_entry = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Update lowest since entry
            if curr_close < lowest_since_entry:
                lowest_since_entry = curr_close
            # Exit conditions:
            # 1. ATR trailing stop: price > lowest_since_entry + 2.5 * ATR
            # 2. Opposite Donchian breakout: price > highest_high (20-period high)
            # 3. Trend filter fails: price > 12h EMA50
            if (curr_close > lowest_since_entry + 2.5 * curr_atr or
                curr_close > curr_highest or
                curr_close > curr_ema_50_12h):
                signals[i] = 0.0
                position = 0
                highest_since_entry = 0.0
                lowest_since_entry = 0.0
            else:
                signals[i] = -0.25
                
        else:  # Flat - look for new entries
            # Long entry: price > highest_high (20-period high) AND price > 12h EMA50 AND volume confirm
            if (curr_close > curr_highest and
                curr_close > curr_ema_50_12h and
                curr_volume_confirm):
                signals[i] = 0.25
                position = 1
                entry_price = curr_close
                highest_since_entry = curr_close
                lowest_since_entry = curr_close
            # Short entry: price < lowest_low (20-period low) AND price < 12h EMA50 AND volume confirm
            elif (curr_close < curr_lowest and
                  curr_close < curr_ema_50_12h and
                  curr_volume_confirm):
                signals[i] = -0.25
                position = -1
                entry_price = curr_close
                highest_since_entry = curr_close
                lowest_since_entry = curr_close
            else:
                signals[i] = 0.0
    
    return signals
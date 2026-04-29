#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Donchian(20) breakout + 1d EMA50 trend filter + volume confirmation + ATR(14) trailing stop
# Donchian channels provide clear structure; 1d EMA50 filters for higher timeframe trend alignment;
# volume confirms breakout strength; ATR-based trailing stop manages risk in both bull and bear markets.
# Target: 20-35 trades/year (80-140 total over 4 years) to balance opportunity and fee drag.
# This version fixes previous trade count issues by tightening volume confirmation and adding
# a minimum holding period to reduce churn.

name = "4h_Donchian20_Breakout_1dEMA50_VolumeSpike_ATRTrail_v2"
timeframe = "4h"
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
    
    # Load HTF data ONCE before loop for 1d calculations
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # Calculate 1d EMA50 for trend filter
    ema_50_1d = pd.Series(df_1d['close'].values).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # Calculate ATR(14) for stoploss and volatility filter
    tr1 = high[1:] - low[1:]
    tr2 = np.abs(high[1:] - close[:-1])
    tr3 = np.abs(low[1:] - close[:-1])
    tr = np.concatenate([[np.nan], np.maximum(tr1, np.maximum(tr2, tr3))])
    atr = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    # Calculate Donchian channels (20-period) - using shifted values to avoid look-ahead
    # Upper channel: highest high of previous 20 periods
    # Lower channel: lowest low of previous 20 periods
    highest_high = pd.Series(high).rolling(window=20, min_periods=20).max().shift(1).values
    lowest_low = pd.Series(low).rolling(window=20, min_periods=20).min().shift(1).values
    
    # Volume confirmation: volume > 2.0x 20-period average (tighter than before)
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_confirm = volume > (2.0 * vol_ma_20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    atr_at_entry = 0.0
    max_high_since_entry = 0.0  # For trailing stop
    min_low_since_entry = 0.0   # For trailing stop
    bars_since_entry = 0        # For minimum holding period
    
    start_idx = max(50, 20, 14)  # warmup for EMA50, Donchian, ATR, volume
    
    for i in range(start_idx, n):
        # Skip if HTF data not available
        if np.isnan(ema_50_1d_aligned[i]) or np.isnan(highest_high[i]) or np.isnan(lowest_low[i]):
            signals[i] = 0.0
            continue
            
        curr_close = close[i]
        curr_high = high[i]
        curr_low = low[i]
        curr_open = open_price[i]
        curr_ema_50_1d = ema_50_1d_aligned[i]
        curr_atr = atr[i]
        curr_upper = highest_high[i]
        curr_lower = lowest_low[i]
        curr_volume_confirm = volume_confirm[i]
        
        # Handle position exits and stops
        if position == 1:  # Long position
            # Update trailing stop: highest high since entry
            max_high_since_entry = max(max_high_since_entry, curr_high)
            # ATR-based trailing stop
            trail_stop = max_high_since_entry - 2.5 * curr_atr
            
            # Exit conditions:
            # 1. Stoploss hit (trailing stop)
            # 2. Price crosses below 1d EMA50 (trend change)
            # 3. Price drops below Donchian lower channel (breakdown)
            # 4. Minimum holding period of 3 bars (12h for 4h timeframe)
            if (curr_low <= trail_stop or
                curr_close < curr_ema_50_1d or
                curr_close < curr_lower or
                bars_since_entry < 3):
                signals[i] = 0.0
                position = 0
                max_high_since_entry = 0.0
                min_low_since_entry = 0.0
                bars_since_entry = 0
            else:
                signals[i] = 0.25
                bars_since_entry += 1
                
        elif position == -1:  # Short position
            # Update trailing stop: lowest low since entry
            min_low_since_entry = min(min_low_since_entry, curr_low)
            # ATR-based trailing stop
            trail_stop = min_low_since_entry + 2.5 * curr_atr
            
            # Exit conditions:
            # 1. Stoploss hit (trailing stop)
            # 2. Price crosses above 1d EMA50 (trend change)
            # 3. Price rises above Donchian upper channel (breakout)
            # 4. Minimum holding period of 3 bars (12h for 4h timeframe)
            if (curr_high >= trail_stop or
                curr_close > curr_ema_50_1d or
                curr_close > curr_upper or
                bars_since_entry < 3):
                signals[i] = 0.0
                position = 0
                max_high_since_entry = 0.0
                min_low_since_entry = 0.0
                bars_since_entry = 0
            else:
                signals[i] = -0.25
                bars_since_entry += 1
                
        else:  # Flat - look for new entries
            # Long entry: price breaks above Donchian upper channel + above 1d EMA50 + volume confirm
            if (curr_close > curr_upper and
                curr_close > curr_ema_50_1d and
                curr_volume_confirm):
                signals[i] = 0.25
                position = 1
                entry_price = curr_close
                atr_at_entry = curr_atr
                max_high_since_entry = curr_high
                min_low_since_entry = curr_low
                bars_since_entry = 0
            # Short entry: price breaks below Donchian lower channel + below 1d EMA50 + volume confirm
            elif (curr_close < curr_lower and
                  curr_close < curr_ema_50_1d and
                  curr_volume_confirm):
                signals[i] = -0.25
                position = -1
                entry_price = curr_close
                atr_at_entry = curr_atr
                max_high_since_entry = curr_high
                min_low_since_entry = curr_low
                bars_since_entry = 0
            else:
                signals[i] = 0.0
                bars_since_entry = 0
    
    return signals
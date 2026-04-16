#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Donchian(20) breakout + weekly pivot direction + volume confirmation
# Long when price breaks above Donchian high + weekly pivot price is above daily open + volume > 1.2x 6h average volume
# Short when price breaks below Donchian low + weekly pivot price is below daily open + volume > 1.2x 6h average volume
# ATR trailing stop (2.5x ATR) for risk management
# Donchian provides trend-following structure, weekly pivot adds higher timeframe bias, volume confirms conviction
# Target: 50-150 total trades over 4 years (12-37/year) to avoid fee drag

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # === 6h Donchian(20) channels ===
    donchian_period = 20
    donchian_high = pd.Series(high).rolling(window=donchian_period, min_periods=donchian_period).max().values
    donchian_low = pd.Series(low).rolling(window=donchian_period, min_periods=donchian_period).min().values
    
    # === 1d Weekly Pivot (from Monday open) ===
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    open_1d = df_1d['open'].values
    
    # Weekly pivot: average of Monday's open, weekly high, weekly low
    # We'll use daily data to approximate: weekly pivot = (weekly high + weekly low + Monday's open) / 3
    # Calculate rolling weekly high/low (5 trading days)
    week_high = pd.Series(high_1d).rolling(window=5, min_periods=5).max().values
    week_low = pd.Series(low_1d).rolling(window=5, min_periods=5).min().values
    # Monday's open: we'll use the first day of the week's open (simplified as current day's open for proxy)
    weekly_pivot = (week_high + week_low + open_1d) / 3
    weekly_pivot_aligned = align_htf_to_ltf(prices, df_1d, weekly_pivot)
    
    # === 6h Volume Confirmation (average volume) ===
    df_6h = get_htf_data(prices, '6h')
    volume_6h = df_6h['volume'].values
    vol_ma_6h = pd.Series(volume_6h).rolling(window=20, min_periods=20).mean().values  # 20 periods average
    vol_ma_6h_aligned = align_htf_to_ltf(prices, df_6h, vol_ma_6h)
    
    # === 6h ATR for trailing stop (14-period) ===
    high_6h_arr = df_6h['high'].values
    low_6h_arr = df_6h['low'].values
    close_6h_arr = df_6h['close'].values
    
    tr1_6h = high_6h_arr - low_6h_arr
    tr2_6h = np.abs(high_6h_arr - np.roll(close_6h_arr, 1))
    tr3_6h = np.abs(low_6h_arr - np.roll(close_6h_arr, 1))
    tr2_6h[0] = tr1_6h[0]
    tr3_6h[0] = tr1_6h[0]
    tr_6h = np.maximum(tr1_6h, np.maximum(tr2_6h, tr3_6h))
    atr_6h = pd.Series(tr_6h).rolling(window=14, min_periods=14).mean().values
    atr_6h_aligned = align_htf_to_ltf(prices, df_6h, atr_6h)
    
    signals = np.zeros(n)
    
    # Warmup
    warmup = 50
    
    # Track position and entry price for trailing stop
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    highest_since_entry = 0.0
    lowest_since_entry = 0.0
    
    for i in range(warmup, n):
        # Skip if any data is NaN
        if (np.isnan(donchian_high[i]) or 
            np.isnan(donchian_low[i]) or
            np.isnan(weekly_pivot_aligned[i]) or
            np.isnan(vol_ma_6h_aligned[i]) or
            np.isnan(atr_6h_aligned[i])):
            signals[i] = 0.0
            position = 0
            continue
        
        price = close[i]
        donch_high = donchian_high[i]
        donch_low = donchian_low[i]
        weekly_pivot_val = weekly_pivot_aligned[i]
        vol_ma_val = vol_ma_6h_aligned[i]
        atr_val = atr_6h_aligned[i]
        daily_open = open_1d[i] if i < len(open_1d) else open_1d[-1]  # approximate daily open
        
        # Volume confirmation: current volume > 1.2x 6h average volume
        vol_confirm = volume[i] > vol_ma_val * 1.2
        
        # Weekly pivot bias: pivot above daily open = bullish bias, below = bearish bias
        weekly_bullish = weekly_pivot_val > daily_open
        weekly_bearish = weekly_pivot_val < daily_open
        
        # === TRAILING STOP LOGIC ===
        if position == 1:  # Long position
            # Update highest price since entry
            if price > highest_since_entry:
                highest_since_entry = price
            # Trail stop: exit if price drops 2.5*ATR from highest
            if atr_val > 0 and price < highest_since_entry - 2.5 * atr_val:
                signals[i] = 0.0
                position = 0
                highest_since_entry = 0.0
                continue
        
        elif position == -1:  # Short position
            # Update lowest price since entry
            if price < lowest_since_entry or lowest_since_entry == 0:
                lowest_since_entry = price
            # Trail stop: exit if price rises 2.5*ATR from lowest
            if atr_val > 0 and price > lowest_since_entry + 2.5 * atr_val:
                signals[i] = 0.0
                position = 0
                lowest_since_entry = 0.0
                continue
        
        # === ENTRY LOGIC (only when flat) ===
        if position == 0:
            # Long when: price breaks above Donchian high AND weekly bullish bias AND volume confirmation
            if price > donch_high and weekly_bullish and vol_confirm:
                signals[i] = 0.25
                position = 1
                entry_price = price
                highest_since_entry = price
                continue
            # Short when: price breaks below Donchian low AND weekly bearish bias AND volume confirmation
            elif price < donch_low and weekly_bearish and vol_confirm:
                signals[i] = -0.25
                position = -1
                entry_price = price
                lowest_since_entry = price
                continue
        
        # Hold current position
        if position == 1:
            signals[i] = 0.25
        elif position == -1:
            signals[i] = -0.25
        else:
            signals[i] = 0.0
    
    return signals

name = "6h_Donchian20_WeeklyPivotBias_Volume1.2x_ATRTrail_2.5x"
timeframe = "6h"
leverage = 1.0
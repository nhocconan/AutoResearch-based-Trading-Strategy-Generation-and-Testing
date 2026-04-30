#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Donchian(20) breakout with weekly pivot direction filter and volume confirmation
# Uses discrete sizing 0.25 to balance profit and fee drag. Target: 50-150 total trades over 4 years (12-37/year).
# Donchian provides structure from 20-bar highs/lows; weekly pivot filters counter-trend moves.
# Volume spike ensures institutional participation. Works in both bull and bear via weekly trend filter.

name = "6h_Donchian20_1wPivot_VolumeSpike_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    open_time = prices['open_time'].values
    
    # Pre-compute session hours (08-20 UTC)
    hours = pd.DatetimeIndex(open_time).hour
    in_session = (hours >= 8) & (hours <= 20)
    
    # Calculate 6h Donchian channels (20-period)
    donchian_h = pd.Series(high).rolling(window=20, min_periods=20).max().values
    donchian_l = pd.Series(low).rolling(window=20, min_periods=20).min().values
    donchian_m = (donchian_h + donchian_l) / 2
    
    # Calculate 1w pivot points (based on prior week's range)
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 1:
        return np.zeros(n)
    
    # Weekly aggregation: get Friday's data for each week
    df_1w_copy = df_1w.copy()
    df_1w_copy['week_end'] = pd.DatetimeIndex(df_1w_copy['open_time']).to_period('W').end_time
    weekly_agg = df_1w_copy.groupby('week_end').agg({
        'high': 'max',
        'low': 'min',
        'close': 'last'
    }).reset_index()
    
    # Map weekly data to 6h bars
    week_end_6h = pd.DatetimeIndex(open_time).to_period('W').end_time
    week_end_vals = weekly_agg.set_index('week_end')
    
    weekly_high = week_end_vals['high'].reindex(week_end_6h).values
    weekly_low = week_end_vals['low'].reindex(week_end_6h).values
    weekly_close = week_end_vals['close'].reindex(week_end_6h).values
    
    # Handle NaN from forward fill (first week)
    weekly_high = pd.Series(weekly_high).ffill().bfill().values
    weekly_low = pd.Series(weekly_low).ffill().bfill().values
    weekly_close = pd.Series(weekly_close).ffill().bfill().values
    
    hl_range = weekly_high - weekly_low
    weekly_pivot = (weekly_high + weekly_low + weekly_close) / 3
    weekly_r1 = 2 * weekly_pivot - weekly_low
    weekly_s1 = 2 * weekly_pivot - weekly_high
    
    # Align weekly pivot to 6h timeframe
    weekly_pivot_aligned = align_htf_to_ltf(prices, df_1w, weekly_pivot)
    weekly_r1_aligned = align_htf_to_ltf(prices, df_1w, weekly_r1)
    weekly_s1_aligned = align_htf_to_ltf(prices, df_1w, weekly_s1)
    
    # Volume confirmation: volume > 2.0x 24-period average (strict to reduce trades)
    vol_ma_24 = pd.Series(volume).rolling(window=24, min_periods=24).mean().values
    volume_spike = volume > (2.0 * vol_ma_24)
    
    # ATR for stoploss (14-period)
    tr1 = high[1:] - low[1:]
    tr2 = np.abs(high[1:] - close[:-1])
    tr3 = np.abs(low[1:] - close[:-1])
    tr = np.concatenate([[np.nan], np.maximum(tr1, np.maximum(tr2, tr3))])
    atr_14 = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    start_idx = max(100, 20, 24, 14)  # warmup
    
    for i in range(start_idx, n):
        # Skip if indicators not ready or outside session
        if (np.isnan(donchian_h[i]) or np.isnan(donchian_l[i]) or
            np.isnan(weekly_pivot_aligned[i]) or np.isnan(weekly_r1_aligned[i]) or
            np.isnan(weekly_s1_aligned[i]) or np.isnan(vol_ma_24[i]) or
            np.isnan(atr_14[i]) or not in_session[i]):
            signals[i] = 0.0
            continue
            
        curr_close = close[i]
        curr_high = high[i]
        curr_low = low[i]
        curr_donchian_h = donchian_h[i]
        curr_donchian_l = donchian_l[i]
        curr_donchian_m = donchian_m[i]
        curr_weekly_pivot = weekly_pivot_aligned[i]
        curr_weekly_r1 = weekly_r1_aligned[i]
        curr_weekly_s1 = weekly_s1_aligned[i]
        curr_volume_spike = volume_spike[i]
        curr_atr = atr_14[i]
        
        if position == 0:  # Flat - look for new entries
            # Only trade on volume spike with Donchian break and weekly pivot filter
            if curr_volume_spike:
                # Bullish: Close breaks above Donchian high + close above weekly pivot
                if curr_close > curr_donchian_h and curr_close > curr_weekly_pivot:
                    signals[i] = 0.25
                    position = 1
                    entry_price = curr_close
                # Bearish: Close breaks below Donchian low + close below weekly pivot
                elif curr_close < curr_donchian_l and curr_close < curr_weekly_pivot:
                    signals[i] = -0.25
                    position = -1
                    entry_price = curr_close
        
        elif position == 1:  # Long position
            # Stoploss: 2 * ATR below entry
            stop_loss = entry_price - 2.0 * curr_atr
            # Exit: Stoploss hit OR close drops below Donchian low OR loses weekly pivot support
            if curr_low <= stop_loss or curr_close < curr_donchian_l or curr_close < curr_weekly_pivot:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:  # Short position
            # Stoploss: 2 * ATR above entry
            stop_loss = entry_price + 2.0 * curr_atr
            # Exit: Stoploss hit OR close rises above Donchian high OR loses weekly pivot resistance
            if curr_high >= stop_loss or curr_close > curr_donchian_h or curr_close > curr_weekly_pivot:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals
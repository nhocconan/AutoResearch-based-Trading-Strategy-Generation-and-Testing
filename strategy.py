#!/usr/bin/env python3
"""
6h_Camarilla_R4_S4_Breakout_1wTrend_VolumeConfirmation
Hypothesis: On 6h timeframe, price breaking Camarilla R4/S4 levels (strong breakout) in the direction of 1w EMA50 trend with volume confirmation (>1.8x 20-period MA) captures high-momentum moves with low false breakouts. The 1w EMA50 acts as a robust trend filter, Camarilla R4/S4 represents extreme levels where breakouts often continue, and volume spike confirms institutional participation. Designed for 12-37 trades/year with discrete sizing (±0.25) and ATR-based trailing stop (2.5x) to minimize fee drag and work in both bull/bear markets with BTC/ETH edge.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Load 1w and 1d data ONCE before loop
    df_1w = get_htf_data(prices, '1w')
    df_1d = get_htf_data(prices, '1d')
    if len(df_1w) < 50 or len(df_1d) < 20:
        return np.zeros(n)
    
    # Calculate Camarilla levels from previous 1d bar
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    daily_range = high_1d - low_1d
    camarilla_r4 = close_1d + daily_range * 1.1 / 2  # R4 level
    camarilla_s4 = close_1d - daily_range * 1.1 / 2  # S4 level
    
    # Align Camarilla levels to 6h timeframe (wait for completed 1d bar)
    camarilla_r4_aligned = align_htf_to_ltf(prices, df_1d, camarilla_r4)
    camarilla_s4_aligned = align_htf_to_ltf(prices, df_1d, camarilla_s4)
    
    # 1w EMA50 for trend filter
    close_1w_series = pd.Series(df_1w['close'])
    ema_50_1w = close_1w_series.ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_50_1w)
    
    # 6h ATR(20) for trailing stop
    tr1 = pd.Series(high).diff().abs()
    tr2 = (pd.Series(high) - pd.Series(close).shift()).abs()
    tr3 = (pd.Series(low) - pd.Series(close).shift()).abs()
    tr_6h = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr_6h = tr_6h.ewm(span=20, adjust=False, min_periods=20).mean()
    atr_6h_values = atr_6h.values
    
    # Volume confirmation: volume > 1.8 * 20-period MA on 6h
    volume_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (volume_ma * 1.8)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    base_size = 0.25
    highest_since_long = 0.0
    lowest_since_short = 0.0
    
    # Warmup: max of 1w EMA (50), ATR (20), volume MA (20) + time for 1d alignment
    start_idx = max(50, 20, 20) + 4  # +4 to ensure 1d bar completion (6h -> 1d: 4 bars per day)
    
    for i in range(start_idx, n):
        close_val = close[i]
        high_val = high[i]
        low_val = low[i]
        vol = volume[i]
        r4_val = camarilla_r4_aligned[i]
        s4_val = camarilla_s4_aligned[i]
        ema_val = ema_50_1w_aligned[i]
        vol_spike = volume_spike[i]
        atr_val = atr_6h_values[i]
        
        # Skip if any data not ready (NaN from alignment or calculation)
        if (np.isnan(r4_val) or np.isnan(s4_val) or np.isnan(ema_val) or 
            np.isnan(atr_val) or np.isnan(volume_ma[i])):
            # Hold current position
            signals[i] = base_size if position == 1 else (-base_size if position == -1 else 0.0)
            continue
        
        # Trend filter: bullish when price > 1w EMA50, bearish when price < 1w EMA50
        trend_bullish = close_val > ema_val
        trend_bearish = close_val < ema_val
        
        # Camarilla R4/S4 breakout conditions: price breaks extreme levels with trend alignment + volume spike
        long_breakout = close_val > r4_val
        short_breakout = close_val < s4_val
        
        long_entry = trend_bullish and long_breakout and vol_spike
        short_entry = trend_bearish and short_breakout and vol_spike
        
        # Update highest/lowest for trailing stop (ATR-based)
        if position == 1:
            highest_since_long = max(highest_since_long, high_val)
        elif position == -1:
            lowest_since_short = min(lowest_since_short, low_val)
        elif position == 0:
            highest_since_long = 0.0
            lowest_since_short = 0.0
        
        # Exit conditions: ATR-based trailing stoploss (wider for 6h)
        long_exit = False
        short_exit = False
        if position == 1:
            # Long trailing stop: highest since entry - 2.5 * ATR
            stop_price = highest_since_long - 2.5 * atr_val
            long_exit = close_val < stop_price
        elif position == -1:
            # Short trailing stop: lowest since entry + 2.5 * ATR
            stop_price = lowest_since_short + 2.5 * atr_val
            short_exit = close_val > stop_price
        
        if long_entry and position != 1:
            signals[i] = base_size
            position = 1
            highest_since_long = high_val
        elif short_entry and position != -1:
            signals[i] = -base_size
            position = -1
            lowest_since_short = low_val
        elif long_exit:
            signals[i] = 0.0
            position = 0
            highest_since_long = 0.0
        elif short_exit:
            signals[i] = 0.0
            position = 0
            lowest_since_short = 0.0
        else:
            # Hold position
            signals[i] = base_size if position == 1 else (-base_size if position == -1 else 0.0)
    
    return signals

name = "6h_Camarilla_R4_S4_Breakout_1wTrend_VolumeConfirmation"
timeframe = "6h"
leverage = 1.0
#!/usr/bin/env python3
"""
4h_Camarilla_R3_S3_Breakout_1dEMA34_Trend_VolumeSpike
Hypothesis: 4h Camarilla R3/S3 breakouts filtered by 1d EMA34 trend and volume spike (volume > 1.8 * 20-period MA) capture strong medium-term trend moves with fewer false signals. Uses ATR trailing stop (2.0x) and discrete position sizing (0.0, ±0.25) to minimize fee churn. Targets 20-40 trades/year on 4h timeframe. Works in both bull and bear markets by following the 1d trend direction only and requiring volume confirmation to avoid false breakouts.
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
    
    # Load 1d data ONCE before loop for EMA34 trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 34:
        return np.zeros(n)
    
    # 1d EMA34 for trend filter
    ema34_1d = pd.Series(df_1d['close']).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema34_1d)
    
    # 4h ATR(10) for stoploss calculation
    tr1 = pd.Series(high).diff().abs()
    tr2 = (pd.Series(high) - pd.Series(close).shift()).abs()
    tr3 = (pd.Series(low) - pd.Series(close).shift()).abs()
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr_4h = tr.ewm(span=10, adjust=False, min_periods=10).mean()
    atr_4h_values = atr_4h.values
    
    # Volume spike filter: volume > 1.8 * 20-period MA on 4h
    volume_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (volume_ma * 1.8)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    base_size = 0.25
    entry_price = 0.0
    highest_since_long = 0.0
    lowest_since_short = 0.0
    
    # Warmup: max of EMA34 (34), ATR (10), volume MA (20)
    start_idx = max(34, 10, 20)
    
    for i in range(start_idx, n):
        close_val = close[i]
        high_val = high[i]
        low_val = low[i]
        vol = volume[i]
        trend_val = ema34_1d_aligned[i]
        atr_val = atr_4h_values[i]
        vol_spike = volume_spike[i]
        
        # Skip if any data not ready
        if (np.isnan(trend_val) or np.isnan(atr_val)):
            # Hold current position
            signals[i] = base_size if position == 1 else (-base_size if position == -1 else 0.0)
            continue
        
        # Trend filter: price > 1d EMA34 = uptrend, price < 1d EMA34 = downtrend
        is_uptrend = close_val > trend_val
        is_downtrend = close_val < trend_val
        
        # Calculate Camarilla levels for previous 4h bar
        if i >= 1:
            # Use previous bar's high, low, close for today's Camarilla levels
            ph = high[i-1]
            pl = low[i-1]
            pc = close[i-1]
            rng = ph - pl
            # Camarilla R3 and S3 levels
            r3 = pc + (rng * 1.1 / 4)
            s3 = pc - (rng * 1.1 / 4)
        else:
            r3 = high_val
            s3 = low_val
        
        # Camarilla breakout conditions
        long_breakout = close_val > r3
        short_breakout = close_val < s3
        
        # Entry conditions: Camarilla breakout in direction of 1d trend + volume spike
        long_entry = long_breakout and is_uptrend and vol_spike
        short_entry = short_breakout and is_downtrend and vol_spike
        
        # Update highest/lowest for trailing stop (ATR-based)
        if position == 1:
            highest_since_long = max(highest_since_long, high_val)
        elif position == -1:
            lowest_since_short = min(lowest_since_short, low_val)
        elif position == 0:
            highest_since_long = 0.0
            lowest_since_short = 0.0
        
        # Exit conditions: ATR-based trailing stoploss (tighter for fewer whipsaws)
        long_exit = False
        short_exit = False
        if position == 1:
            # Long trailing stop: highest since entry - 2.0 * ATR
            stop_price = highest_since_long - 2.0 * atr_val
            long_exit = close_val < stop_price
        elif position == -1:
            # Short trailing stop: lowest since entry + 2.0 * ATR
            stop_price = lowest_since_short + 2.0 * atr_val
            short_exit = close_val > stop_price
        
        if long_entry and position != 1:
            signals[i] = base_size
            position = 1
            entry_price = close_val
            highest_since_long = high_val
        elif short_entry and position != -1:
            signals[i] = -base_size
            position = -1
            entry_price = close_val
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

name = "4h_Camarilla_R3_S3_Breakout_1dEMA34_Trend_VolumeSpike"
timeframe = "4h"
leverage = 1.0
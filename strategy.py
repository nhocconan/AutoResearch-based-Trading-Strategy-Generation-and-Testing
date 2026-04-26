#!/usr/bin/env python3
"""
1h_VWAP_Reversion_4hTrend
Hypothesis: In 1h timeframe, price tends to revert to VWAP during strong 4h trends. 
Enter long when price crosses below VWAP in 4h uptrend, short when above VWAP in 4h downtrend.
Exit when price returns to VWAP or 4h trend reverses. Uses session filter (08-20 UTC) 
to avoid low-volume periods. Targets 15-35 trades/year with discrete sizing (0.0, ±0.20) 
to minimize fee drag. Works in bull/bear by following 4h trend direction only.
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
    
    # Load 4h data ONCE before loop for trend and VWAP reference
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 20:
        return np.zeros(n)
    
    # 4h VWAP (typical price * volume cumsum)
    typical_price_4h = (df_4h['high'] + df_4h['low'] + df_4h['close']) / 3.0
    vwap_4h = (typical_price_4h * df_4h['volume']).cumsum() / df_4h['volume'].cumsum()
    vwap_4h_values = vwap_4h.values
    
    # 4h EMA20 for trend filter
    ema20_4h = pd.Series(df_4h['close']).ewm(span=20, adjust=False, min_periods=20).mean().values
    
    # Align HTF indicators to 1h timeframe (completed 4h bar only)
    vwap_4h_aligned = align_htf_to_ltf(prices, df_4h, vwap_4h_values)
    ema20_4h_aligned = align_htf_to_ltf(prices, df_4h, ema20_4h)
    
    # 1h VWAP for entry timing
    typical_price = (high + low + close) / 3.0
    vwap = pd.Series(typical_price * volume).cumsum() / pd.Series(volume).cumsum()
    vwap_values = vwap.values
    
    # Session filter: 08-20 UTC
    hours = prices.index.hour
    in_session = (hours >= 8) & (hours <= 20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    base_size = 0.20
    
    # Warmup: max of 4h EMA20 (20), 1h VWAP needs volume
    start_idx = 20
    
    for i in range(start_idx, n):
        # Skip if outside trading session
        if not in_session[i]:
            # Hold current position or flat
            signals[i] = base_size if position == 1 else (-base_size if position == -1 else 0.0)
            continue
            
        close_val = close[i]
        vwap_1h_val = vwap_values[i]
        vwap_4h_val = vwap_4h_aligned[i]
        ema20_4h_val = ema20_4h_aligned[i]
        
        # Skip if any data not ready
        if (np.isnan(vwap_4h_val) or np.isnan(ema20_4h_val) or np.isnan(vwap_1h_val)):
            signals[i] = base_size if position == 1 else (-base_size if position == -1 else 0.0)
            continue
        
        # 4h trend filter: price > EMA20 = uptrend, price < EMA20 = downtrend
        is_uptrend = close_val > ema20_4h_val
        is_downtrend = close_val < ema20_4h_val
        
        # 1h VWAP reversion signals
        vwap_dev_1h = close_val - vwap_1h_val  # positive = above VWAP
        
        # Entry: price deviates from 1h VWAP in opposite direction of 4h trend
        # In 4h uptrend, look for 1h price below VWAP (mean reversion long)
        # In 4h downtrend, look for 1h price above VWAP (mean reversion short)
        long_entry = is_uptrend and (vwap_dev_1h < -0.001)  # slightly below VWAP
        short_entry = is_downtrend and (vwap_dev_1h > 0.001)  # slightly above VWAP
        
        # Exit: price returns to 1h VWAP or 4h trend reverses
        long_exit = (vwap_dev_1h > -0.0005) or not is_uptrend  # near VWAP or trend change
        short_exit = (vwap_dev_1h < 0.0005) or not is_downtrend  # near VWAP or trend change
        
        if long_entry and position != 1:
            signals[i] = base_size
            position = 1
        elif short_entry and position != -1:
            signals[i] = -base_size
            position = -1
        elif long_exit and position == 1:
            signals[i] = 0.0
            position = 0
        elif short_exit and position == -1:
            signals[i] = 0.0
            position = 0
        else:
            # Hold position
            signals[i] = base_size if position == 1 else (-base_size if position == -1 else 0.0)
    
    return signals

name = "1h_VWAP_Reversion_4hTrend"
timeframe = "1h"
leverage = 1.0
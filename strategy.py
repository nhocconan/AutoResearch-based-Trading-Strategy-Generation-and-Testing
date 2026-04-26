#!/usr/bin/env python3
"""
1h_4h1d_Camarilla_R1S1_Breakout_Volume_Trend_v1
Hypothesis: Camarilla R1/S1 breakouts on 1h timeframe with 4h/1d trend alignment and volume confirmation capture institutional breakout moves while avoiding false signals in choppy markets. Uses discrete sizing (0.20) to target 15-37 trades/year. Works in bull/bear by taking breakouts in direction of higher-timeframe trend (4h/1d EMA34). Volume confirmation (>1.8x 20-bar average) ensures momentum validity. Designed for low trade frequency to minimize fee drag on 1h timeframe.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Load HTF data ONCE before loop
    df_4h = get_htf_data(prices, '4h')
    df_1d = get_htf_data(prices, '1d')
    
    if len(df_4h) < 34 or len(df_1d) < 34:
        return np.zeros(n)
    
    # 4h and 1d EMA34 for trend filter
    ema34_4h = pd.Series(df_4h['close']).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema34_1d = pd.Series(df_1d['close']).ewm(span=34, adjust=False, min_periods=34).mean().values
    
    ema34_4h_aligned = align_htf_to_ltf(prices, df_4h, ema34_4h)
    ema34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema34_1d)
    
    # Volume confirmation: current volume > 1.8 * 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_confirm = volume > (vol_ma * 1.8)
    
    # Session filter: 08-20 UTC
    hours = prices.index.hour
    session_filter = (hours >= 8) & (hours <= 20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    base_size = 0.20
    
    # Warmup: max of EMA34 (34), volume MA (20)
    start_idx = max(34, 20)
    
    for i in range(start_idx, n):
        close_val = close[i]
        high_val = high[i]
        low_val = low[i]
        trend_4h = ema34_4h_aligned[i]
        trend_1d = ema34_1d_aligned[i]
        vol_conf = volume_confirm[i]
        in_session = session_filter[i]
        
        # Skip if any data not ready or outside session
        if (np.isnan(trend_4h) or np.isnan(trend_1d) or not in_session):
            signals[i] = base_size if position == 1 else (-base_size if position == -1 else 0.0)
            continue
        
        # Calculate Camarilla levels for previous 1h period (using prior bar's OHLC)
        if i >= 1:
            prev_close = close[i-1]
            prev_high = high[i-1]
            prev_low = low[i-1]
            cam_range = prev_high - prev_low
            
            # Camarilla R1, S1 levels
            r1 = prev_close + cam_range * 1.1 / 12
            s1 = prev_close - cam_range * 1.1 / 12
        else:
            r1 = close_val
            s1 = close_val
        
        # Trend filter: price > both 4h and 1d EMA34 = uptrend, price < both = downtrend
        is_uptrend = close_val > trend_4h and close_val > trend_1d
        is_downtrend = close_val < trend_4h and close_val < trend_1d
        
        # Camarilla breakout conditions
        long_breakout = close_val > r1
        short_breakout = close_val < s1
        
        # Entry conditions: Camarilla breakout in direction of 4h/1d trend + volume
        long_entry = long_breakout and is_uptrend and vol_conf
        short_entry = short_breakout and is_downtrend and vol_conf
        
        if long_entry and position != 1:
            signals[i] = base_size
            position = 1
        elif short_entry and position != -1:
            signals[i] = -base_size
            position = -1
        elif position == 1 and close_val < s1:  # Long exit: price breaks below S1
            signals[i] = 0.0
            position = 0
        elif position == -1 and close_val > r1:  # Short exit: price breaks above R1
            signals[i] = 0.0
            position = 0
        else:
            # Hold position
            signals[i] = base_size if position == 1 else (-base_size if position == -1 else 0.0)
    
    return signals

name = "1h_4h1d_Camarilla_R1S1_Breakout_Volume_Trend_v1"
timeframe = "1h"
leverage = 1.0
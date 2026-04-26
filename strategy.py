#!/usr/bin/env python3
"""
1h_Camarilla_R1S1_Breakout_4hTrend_VolumeSpike_v1
Hypothesis: Camarilla R1/S1 breakout on 1h with 4h trend filter and volume spike captures intraday momentum in both bull/bear markets. Uses 4h for signal direction (reducing whipsaw) and 1h for precise entry timing. Discrete sizing (0.20) and strict volume confirmation (2.0x) target 15-37 trades/year. Session filter (08-20 UTC) avoids low-liquidity hours.
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
    
    # Pre-compute session hours filter (08-20 UTC)
    hours = prices.index.hour  # prices.index is DatetimeIndex
    in_session = (hours >= 8) & (hours <= 20)
    
    # Load 4h data ONCE before loop for trend filter and Camarilla levels
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 50:
        return np.zeros(n)
    
    # 4h EMA34 for trend filter
    ema34_4h = pd.Series(df_4h['close']).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema34_4h_aligned = align_htf_to_ltf(prices, df_4h, ema34_4h)
    
    # Previous 4h bar's OHLC for Camarilla calculation
    prev_close = df_4h['close'].shift(1).values
    prev_high = df_4h['high'].shift(1).values
    prev_low = df_4h['low'].shift(1).values
    
    # Camarilla levels: R1 = close + (high-low)*1.1/12, S1 = close - (high-low)*1.1/12
    camarilla_range = prev_high - prev_low
    r1 = prev_close + camarilla_range * 1.1 / 12
    s1 = prev_close - camarilla_range * 1.1 / 12
    
    # Align Camarilla levels to 1h timeframe (using completed 4h bars only)
    r1_aligned = align_htf_to_ltf(prices, df_4h, r1)
    s1_aligned = align_htf_to_ltf(prices, df_4h, s1)
    
    # Volume confirmation: current volume > 2.0 * 20-period average (stricter to reduce trades)
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_confirm = volume > (vol_ma * 2.0)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    base_size = 0.20
    
    # Warmup: max of EMA34 (34), volume MA (20)
    start_idx = max(34, 20)
    
    for i in range(start_idx, n):
        # Skip if outside trading session
        if not in_session[i]:
            signals[i] = 0.0
            position = 0
            continue
            
        close_val = close[i]
        r1_val = r1_aligned[i]
        s1_val = s1_aligned[i]
        trend_val = ema34_4h_aligned[i]
        vol_conf = volume_confirm[i]
        
        # Skip if any data not ready
        if (np.isnan(r1_val) or np.isnan(s1_val) or np.isnan(trend_val)):
            # Hold current position (should be flat during warmup)
            signals[i] = base_size if position == 1 else (-base_size if position == -1 else 0.0)
            continue
        
        # Trend filter: price > EMA34 = uptrend, price < EMA34 = downtrend
        is_uptrend = close_val > trend_val
        is_downtrend = close_val < trend_val
        
        # Entry conditions: Camarilla breakout in direction of trend + volume
        long_condition = (close_val > r1_val) and is_uptrend and vol_conf
        short_condition = (close_val < s1_val) and is_downtrend and vol_conf
        
        # Exit conditions: opposite Camarilla level touch or trend reversal
        long_exit = (position == 1 and (close_val < s1_val or not is_uptrend))
        short_exit = (position == -1 and (close_val > r1_val or not is_downtrend))
        
        if long_condition and position != 1:
            signals[i] = base_size
            position = 1
        elif short_condition and position != -1:
            signals[i] = -base_size
            position = -1
        elif long_exit:
            signals[i] = 0.0
            position = 0
        elif short_exit:
            signals[i] = 0.0
            position = 0
        else:
            # Hold position
            signals[i] = base_size if position == 1 else (-base_size if position == -1 else 0.0)
    
    return signals

name = "1h_Camarilla_R1S1_Breakout_4hTrend_VolumeSpike_v1"
timeframe = "1h"
leverage = 1.0
#!/usr/bin/env python3
"""
1h_Camarilla_R1S1_Breakout_4hTrend_VolumeFilter
Hypothesis: On 1h timeframe, Camarilla R1/S1 breakout with 4h EMA20 trend filter and volume confirmation captures momentum moves with controlled frequency. Uses 4h trend for direction alignment (works in bull/bear) and volume spike to avoid false breakouts. Target: 15-35 trades/year via tight entry conditions and session filter (08-20 UTC).
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
    
    # Load 4h data ONCE before loop for trend filter
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 2:
        return np.zeros(n)
    
    # 4h EMA20 for trend filter
    ema_20_4h = pd.Series(df_4h['close']).ewm(span=20, adjust=False, min_periods=20).mean().values
    ema_20_4h_aligned = align_htf_to_ltf(prices, df_4h, ema_20_4h)
    
    # Previous 4h bar's Camarilla R1/S1 levels (using 4h OHLC)
    prev_close = df_4h['close'].shift(1).values
    prev_high = df_4h['high'].shift(1).values
    prev_low = df_4h['low'].shift(1).values
    
    # Camarilla calculations: R1 = C + (H-L)*1.1/12, S1 = C - (H-L)*1.1/12
    R1 = prev_close + (prev_high - prev_low) * 1.1 / 12
    S1 = prev_close - (prev_high - prev_low) * 1.1 / 12
    
    # Align to 1h timeframe
    R1_aligned = align_htf_to_ltf(prices, df_4h, R1)
    S1_aligned = align_htf_to_ltf(prices, df_4h, S1)
    
    # Volume spike: current volume > 1.8 * 24-period average (≈1 day on 1h)
    vol_ma = pd.Series(volume).rolling(window=24, min_periods=24).mean().values
    volume_spike = volume > (vol_ma * 1.8)
    
    # Session filter: 08-20 UTC (precompute hours array)
    hours = prices.index.hour  # prices.index is DatetimeIndex
    in_session = (hours >= 8) & (hours <= 20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    base_size = 0.20
    
    # Warmup: max of EMA20 (20) and volume MA (24)
    start_idx = max(20, 24)
    
    for i in range(start_idx, n):
        # Skip if outside session
        if not in_session[i]:
            signals[i] = 0.0
            position = 0
            continue
            
        close_val = close[i]
        ema_val = ema_20_4h_aligned[i]
        r1_val = R1_aligned[i]
        s1_val = S1_aligned[i]
        vol_spike = volume_spike[i]
        
        # Skip if any data not ready
        if (np.isnan(ema_val) or np.isnan(r1_val) or np.isnan(s1_val)):
            # Hold current position
            signals[i] = base_size if position == 1 else (-base_size if position == -1 else 0.0)
            continue
        
        # Trend filter: price vs 4h EMA20
        uptrend = close_val > ema_val
        downtrend = close_val < ema_val
        
        # Long: price breaks above R1 with 4h uptrend and volume spike
        long_condition = (close_val > r1_val) and uptrend and vol_spike
        # Short: price breaks below S1 with 4h downtrend and volume spike
        short_condition = (close_val < s1_val) and downtrend and vol_spike
        
        # Exit: price re-enters R1-S1 range
        long_exit = (position == 1 and close_val < r1_val)
        short_exit = (position == -1 and close_val > s1_val)
        
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

name = "1h_Camarilla_R1S1_Breakout_4hTrend_VolumeFilter"
timeframe = "1h"
leverage = 1.0
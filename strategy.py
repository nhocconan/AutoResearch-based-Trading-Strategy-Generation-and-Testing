#!/usr/bin/env python3
"""
1d_Camarilla_R1S1_Breakout_1wTrend_VolumeSpike_v1
Hypothesis: On 1d timeframe, trade Camarilla R1/S1 breakouts with 1w EMA34 trend filter and volume spike confirmation. Uses 1w EMA for long-term trend alignment and volume spike for institutional participation. Designed for 30-100 total trades over 4 years (7-25/year) with discrete sizing (0.25) to minimize fee drag. Works in bull/bear markets via 1w trend filter.
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
    
    # Get 1w data for EMA(34) trend filter and Camarilla pivot calculation
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 1:
        return np.zeros(n)
    
    # Calculate 1w EMA(34) for trend filter
    ema_34_1w = pd.Series(df_1w['close'].values).ewm(span=34, adjust=False, min_periods=34).mean().values
    
    # Calculate Camarilla levels from previous 1w bar
    prev_close = df_1w['close'].values
    prev_high = df_1w['high'].values
    prev_low = df_1w['low'].values
    
    # Typical price for pivot calculation
    typical_price = (prev_high + prev_low + prev_close) / 3.0
    range_hl = prev_high - prev_low
    
    # Camarilla R1 and S1 levels (primary breakout levels)
    r1 = typical_price + range_hl * 1.1 / 4.0
    s1 = typical_price - range_hl * 1.1 / 4.0
    
    # Volume spike: current volume > 2.0 * 20-period average volume
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (2.0 * vol_ma)
    
    # Align HTF indicators to 1d timeframe
    ema_34_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_34_1w)
    r1_aligned = align_htf_to_ltf(prices, df_1w, r1)
    s1_aligned = align_htf_to_ltf(prices, df_1w, s1)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    # Warmup: max of EMA(34) 1w, volume MA (20)
    start_idx = max(34, 20) + 1
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(ema_34_1w_aligned[i]) or 
            np.isnan(r1_aligned[i]) or
            np.isnan(s1_aligned[i])):
            # Hold current position
            if position == 0:
                signals[i] = 0.0
            elif position == 1:
                signals[i] = 0.25
            else:
                signals[i] = -0.25
            continue
        
        ema_34_1w_val = ema_34_1w_aligned[i]
        close_val = close[i]
        high_val = high[i]
        low_val = low[i]
        vol_spike = volume_spike[i]
        r1_val = r1_aligned[i]
        s1_val = s1_aligned[i]
        
        # Trend filter: price > EMA34 (uptrend) or < EMA34 (downtrend)
        uptrend = close_val > ema_34_1w_val
        downtrend = close_val < ema_34_1w_val
        
        if position == 0:
            # Long: break above R1 with uptrend and volume spike
            # Short: break below S1 with downtrend and volume spike
            long_signal = (high_val > r1_val and uptrend and vol_spike)
            short_signal = (low_val < s1_val and downtrend and vol_spike)
            
            if long_signal:
                signals[i] = 0.25
                position = 1
                entry_price = close_val
            elif short_signal:
                signals[i] = -0.25
                position = -1
                entry_price = close_val
            else:
                signals[i] = 0.0
        elif position == 1:
            # Hold long
            signals[i] = 0.25
            # Exit: trend reversal or price reaches S1 (mean reversion target)
            if close_val < ema_34_1w_val or low_val <= s1_val:
                signals[i] = 0.0
                position = 0
        elif position == -1:
            # Hold short
            signals[i] = -0.25
            # Exit: trend reversal or price reaches R1 (mean reversion target)
            if close_val > ema_34_1w_val or high_val >= r1_val:
                signals[i] = 0.0
                position = 0
    
    return signals

name = "1d_Camarilla_R1S1_Breakout_1wTrend_VolumeSpike_v1"
timeframe = "1d"
leverage = 1.0
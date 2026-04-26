#!/usr/bin/env python3
"""
1h_Camarilla_R1S1_Breakout_4hTrend_1dVolSpike_v1
Hypothesis: On 1h timeframe, trade Camarilla R1/S1 breakouts with 4h EMA20 trend filter and 1d volume spike confirmation. 
Camarilla levels provide intraday support/resistance, 4h EMA20 filters intermediate trend to avoid counter-trend trades, 
and 1d volume spike ensures institutional participation. Designed for 60-150 total trades over 4 years (15-37/year) 
with discrete sizing (0.20) to minimize fee drag. Works in bull/bear markets via trend filter and avoids low-volume 
false breakouts by requiring 1d volume confirmation.
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
    
    # Get 4h data for EMA20 trend filter
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 20:
        return np.zeros(n)
    
    # Calculate 4h EMA(20) for trend filter
    ema_20_4h = pd.Series(df_4h['close']).ewm(span=20, adjust=False, min_periods=20).mean().values
    ema_20_4h_aligned = align_htf_to_ltf(prices, df_4h, ema_20_4h)
    
    # Get 1d data for Camarilla pivot calculation and volume spike
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 1:
        return np.zeros(n)
    
    # Calculate Camarilla levels from previous 1d bar
    # Camarilla: based on previous day's range
    prev_close = df_1d['close'].values
    prev_high = df_1d['high'].values
    prev_low = df_1d['low'].values
    
    # Typical price for pivot calculation
    typical_price = (prev_high + prev_low + prev_close) / 3.0
    range_hl = prev_high - prev_low
    
    # Camarilla levels (R1, S1 are the primary breakout levels)
    r1 = typical_price + range_hl * 1.1 / 12.0
    s1 = typical_price - range_hl * 1.1 / 12.0
    r2 = typical_price + range_hl * 1.1 / 6.0
    s2 = typical_price - range_hl * 1.1 / 6.0
    
    # Get 1d EMA(34) for additional trend filter (optional)
    ema_34_1d = pd.Series(prev_close).ewm(span=34, adjust=False, min_periods=34).mean().values
    
    # Volume spike: current 1d volume > 2.0 * 20-period average volume
    vol_ma_1d = pd.Series(df_1d['volume']).rolling(window=20, min_periods=20).mean().values
    volume_spike_1d = df_1d['volume'].values > (2.0 * vol_ma_1d)
    
    # Align HTF indicators to 1h timeframe
    r1_aligned = align_htf_to_ltf(prices, df_1d, r1)
    s1_aligned = align_htf_to_ltf(prices, df_1d, s1)
    r2_aligned = align_htf_to_ltf(prices, df_1d, r2)
    s2_aligned = align_htf_to_ltf(prices, df_1d, s2)
    ema_20_4h_aligned = align_htf_to_ltf(prices, df_4h, ema_20_4h)
    ema_34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    volume_spike_1d_aligned = align_htf_to_ltf(prices, df_1d, volume_spike_1d.astype(float))
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Warmup: max of EMA(20) 4h, EMA(34) 1d, volume MA (20)
    start_idx = max(20, 34, 20) + 1
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(ema_20_4h_aligned[i]) or 
            np.isnan(r1_aligned[i]) or
            np.isnan(s1_aligned[i]) or
            np.isnan(ema_34_1d_aligned[i]) or
            np.isnan(volume_spike_1d_aligned[i])):
            # Hold current position
            if position == 0:
                signals[i] = 0.0
            elif position == 1:
                signals[i] = 0.20
            else:
                signals[i] = -0.20
            continue
        
        ema_20_4h_val = ema_20_4h_aligned[i]
        ema_34_1d_val = ema_34_1d_aligned[i]
        close_val = close[i]
        high_val = high[i]
        low_val = low[i]
        vol_spike_1d = volume_spike_1d_aligned[i] > 0.5
        r1_val = r1_aligned[i]
        s1_val = s1_aligned[i]
        r2_val = r2_aligned[i]
        s2_val = s2_aligned[i]
        
        # Trend filter: price > EMA20_4h (uptrend) or < EMA20_4h (downtrend)
        # Also require alignment with 1d EMA34 for stronger trend confirmation
        uptrend = close_val > ema_20_4h_val and close_val > ema_34_1d_val
        downtrend = close_val < ema_20_4h_val and close_val < ema_34_1d_val
        
        if position == 0:
            # Long: break above R1 with uptrend and 1d volume spike
            long_signal = (high_val > r1_val) and uptrend and vol_spike_1d
            
            # Short: break below S1 with downtrend and 1d volume spike
            short_signal = (low_val < s1_val) and downtrend and vol_spike_1d
            
            if long_signal:
                signals[i] = 0.20
                position = 1
            elif short_signal:
                signals[i] = -0.20
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # Hold long
            signals[i] = 0.20
            # Exit: trend reversal (price < EMA20_4h) or price reaches S1 (mean reversion)
            if close_val < ema_20_4h_val or low_val <= s1_val:
                signals[i] = 0.0
                position = 0
        elif position == -1:
            # Hold short
            signals[i] = -0.20
            # Exit: trend reversal (price > EMA20_4h) or price reaches R1 (mean reversion)
            if close_val > ema_20_4h_val or high_val >= r1_val:
                signals[i] = 0.0
                position = 0
    
    return signals

name = "1h_Camarilla_R1S1_Breakout_4hTrend_1dVolSpike_v1"
timeframe = "1h"
leverage = 1.0
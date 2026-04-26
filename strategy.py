#!/usr/bin/env python3
"""
4h_Camarilla_R1_S1_Breakout_12hTrend_VolumeSpike_v1
Hypothesis: On 4h timeframe, trade Camarilla R1/S1 breakouts with 12h EMA50 trend filter and volume spike confirmation. 
Camarilla levels provide institutional support/resistance, 12h EMA50 filters medium-term trend, and volume spike confirms 
institutional participation. Designed for 75-200 total trades over 4 years (19-50/year) with discrete sizing (0.25) to 
minimize fee drag. Works in bull/bear markets via trend filter and avoids low-volume false breakouts.
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
    
    # Get 12h data for EMA trend filter
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 1:
        return np.zeros(n)
    
    # Get 1d data for Camarilla pivot calculation (using previous day's OHLC)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 1:
        return np.zeros(n)
    
    # Calculate Camarilla levels from previous 1d bar
    prev_close = df_1d['close'].values
    prev_high = df_1d['high'].values
    prev_low = df_1d['low'].values
    
    typical_price = (prev_high + prev_low + prev_close) / 3.0
    range_hl = prev_high - prev_low
    
    # Camarilla R1 and S1 levels (inner levels)
    r1 = typical_price + range_hl * 1.1 / 12.0
    s1 = typical_price - range_hl * 1.1 / 12.0
    r2 = typical_price + range_hl * 1.1 / 6.0
    s2 = typical_price - range_hl * 1.1 / 6.0
    
    # Get 12h EMA(50) for trend filter
    ema_50_12h = pd.Series(df_12h['close'].values).ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Volume spike: current volume > 2.0 * 20-period average volume
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (2.0 * vol_ma)
    
    # Align HTF indicators to 4h timeframe
    r1_aligned = align_htf_to_ltf(prices, df_1d, r1)
    s1_aligned = align_htf_to_ltf(prices, df_1d, s1)
    r2_aligned = align_htf_to_ltf(prices, df_1d, r2)
    s2_aligned = align_htf_to_ltf(prices, df_1d, s2)
    ema_50_12h_aligned = align_htf_to_ltf(prices, df_12h, ema_50_12h)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    # Warmup: max of EMA(50) 12h, volume MA (20)
    start_idx = max(50, 20) + 1
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(ema_50_12h_aligned[i]) or 
            np.isnan(r1_aligned[i]) or
            np.isnan(s1_aligned[i]) or
            np.isnan(r2_aligned[i]) or
            np.isnan(s2_aligned[i])):
            # Hold current position
            if position == 0:
                signals[i] = 0.0
            elif position == 1:
                signals[i] = 0.25
            else:
                signals[i] = -0.25
            continue
        
        ema_50_12h_val = ema_50_12h_aligned[i]
        close_val = close[i]
        high_val = high[i]
        low_val = low[i]
        vol_spike = volume_spike[i]
        r1_val = r1_aligned[i]
        s1_val = s1_aligned[i]
        r2_val = r2_aligned[i]
        s2_val = s2_aligned[i]
        
        # Trend filter: price > EMA50 (uptrend) or < EMA50 (downtrend)
        uptrend = close_val > ema_50_12h_val
        downtrend = close_val < ema_50_12h_val
        
        if position == 0:
            # Long: break above R1 with uptrend and volume spike
            # OR break above R2 with volume spike (stronger signal)
            long_signal = ((high_val > r1_val and uptrend) or (high_val > r2_val)) and vol_spike
            
            # Short: break below S1 with downtrend and volume spike
            # OR break below S2 with volume spike (stronger signal)
            short_signal = ((low_val < s1_val and downtrend) or (low_val < s2_val)) and vol_spike
            
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
            # Exit: trend reversal or price reaches S2 (mean reversion target)
            if close_val < ema_50_12h_val or low_val <= s2_val:
                signals[i] = 0.0
                position = 0
        elif position == -1:
            # Hold short
            signals[i] = -0.25
            # Exit: trend reversal or price reaches R2 (mean reversion target)
            if close_val > ema_50_12h_val or high_val >= r2_val:
                signals[i] = 0.0
                position = 0
    
    return signals

name = "4h_Camarilla_R1_S1_Breakout_12hTrend_VolumeSpike_v1"
timeframe = "4h"
leverage = 1.0
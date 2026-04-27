#!/usr/bin/env python3
"""
12h_Camarilla_R1_S1_Breakout_1dTrend_VolumeSpike_HTF
Hypothesis: Camarilla R1/S1 breakout on 12h with 1d EMA34 trend filter and volume confirmation.
Long when price breaks above R1 with 1d uptrend and volume spike.
Short when price breaks below S1 with 1d downtrend and volume spike.
Camarilla levels provide high-probability reversal/breakout points from prior day's range.
Designed for 12-30 trades/year on 12h to minimize fee drag while maintaining edge.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    open_price = prices['open'].values
    
    # Calculate 12h Camarilla levels from previous 1d candle
    # R1 = close + 1.1*(high-low)/12, S1 = close - 1.1*(high-low)/12
    # Using 1d high/low/close to calculate levels for next 12h period
    df_1d = get_htf_data(prices, '1d')
    h_1d = df_1d['high'].values
    l_1d = df_1d['low'].values
    c_1d = df_1d['close'].values
    
    camarilla_width = 1.1 * (h_1d - l_1d) / 12
    r1 = c_1d + camarilla_width
    s1 = c_1d - camarilla_width
    
    # Align Camarilla levels to 12h timeframe (levels from previous 1d candle)
    r1_aligned = align_htf_to_ltf(prices, df_1d, r1)
    s1_aligned = align_htf_to_ltf(prices, df_1d, s1)
    
    # 1d EMA34 for trend filter
    ema_34_1d = pd.Series(df_1d['close'].values).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # Volume spike: current volume > 2.0 * 20-period average
    vol_avg = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (2.0 * vol_avg)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    size = 0.25  # 25% position size
    
    # Warmup: need enough for 1d EMA34 and volume average
    start_idx = max(34, 20)
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(r1_aligned[i]) or np.isnan(s1_aligned[i]) or
            np.isnan(ema_34_1d_aligned[i]) or np.isnan(volume_spike[i])):
            signals[i] = 0.0
            continue
        
        close_val = close[i]
        open_val = open_price[i]
        ema_trend = ema_34_1d_aligned[i]
        vol_spike = volume_spike[i]
        r1_level = r1_aligned[i]
        s1_level = s1_aligned[i]
        
        if position == 0:
            # Flat - look for breakout entry with trend and volume confirmation
            # Long: price breaks above R1 AND 1d trend up (close > EMA34) AND volume spike
            # Short: price breaks below S1 AND 1d trend down (close < EMA34) AND volume spike
            long_condition = close_val > r1_level and open_val <= r1_level and close_val > ema_trend and vol_spike
            short_condition = close_val < s1_level and open_val >= s1_level and close_val < ema_trend and vol_spike
            
            if long_condition:
                signals[i] = size
                position = 1
                entry_price = close_val
            elif short_condition:
                signals[i] = -size
                position = -1
                entry_price = close_val
        elif position == 1:
            # Long - exit when price breaks below S1 (reversal) OR 1d trend turns down
            if close_val < s1_level or close_val < ema_trend:
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
            else:
                signals[i] = size
        elif position == -1:
            # Short - exit when price breaks above R1 (reversal) OR 1d trend turns up
            if close_val > r1_level or close_val > ema_trend:
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
            else:
                signals[i] = -size
    
    return signals

name = "12h_Camarilla_R1_S1_Breakout_1dTrend_VolumeSpike_HTF"
timeframe = "12h"
leverage = 1.0
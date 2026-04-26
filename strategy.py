#!/usr/bin/env python3
"""
4h_Camarilla_R1S1_Breakout_1dEMA34_Trend_ATRStop_v2
Hypothesis: On 4h timeframe, trade Camarilla R1/S1 breakouts with 1d EMA34 trend filter and ATR-based stoploss. Uses 1d EMA for long-term trend alignment (works in bull/bear) and ATR stop for risk control. Designed for 75-200 total trades over 4 years with discrete sizing (0.25) to minimize fee drag. Volume confirmation ensures institutional participation.
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
    
    # Get 1d data for EMA(34) trend filter and Camarilla pivot calculation
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 1:
        return np.zeros(n)
    
    # Calculate 1d EMA(34) for trend filter
    ema_34_1d = pd.Series(df_1d['close'].values).ewm(span=34, adjust=False, min_periods=34).mean().values
    
    # Calculate ATR(14) for stoploss
    tr1 = np.abs(high - low)
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr1[0] = 0
    tr2[0] = 0
    tr3[0] = 0
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    atr = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    # Calculate Camarilla levels from previous 1d bar
    prev_close = df_1d['close'].values
    prev_high = df_1d['high'].values
    prev_low = df_1d['low'].values
    
    # Typical price for pivot calculation
    typical_price = (prev_high + prev_low + prev_close) / 3.0
    range_hl = prev_high - prev_low
    
    # Camarilla R1 and S1 levels (primary breakout levels)
    r1 = typical_price + range_hl * 1.1 / 4.0
    s1 = typical_price - range_hl * 1.1 / 4.0
    
    # Volume spike: current volume > 2.0 * 20-period average volume
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (2.0 * vol_ma)
    
    # Align HTF indicators to 4h timeframe
    ema_34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    r1_aligned = align_htf_to_ltf(prices, df_1d, r1)
    s1_aligned = align_htf_to_ltf(prices, df_1d, s1)
    atr_aligned = align_htf_to_ltf(prices, df_1d, atr)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    max_high_since_entry = 0.0
    min_low_since_entry = 0.0
    
    # Warmup: max of EMA(34) 1d, ATR(14), volume MA (20)
    start_idx = max(34, 14, 20) + 1
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(ema_34_1d_aligned[i]) or 
            np.isnan(r1_aligned[i]) or
            np.isnan(s1_aligned[i]) or
            np.isnan(atr_aligned[i])):
            # Hold current position
            if position == 0:
                signals[i] = 0.0
            elif position == 1:
                signals[i] = 0.25
            else:
                signals[i] = -0.25
            continue
        
        ema_34_1d_val = ema_34_1d_aligned[i]
        close_val = close[i]
        high_val = high[i]
        low_val = low[i]
        vol_spike = volume_spike[i]
        r1_val = r1_aligned[i]
        s1_val = s1_aligned[i]
        atr_val = atr_aligned[i]
        
        # Trend filter: price > EMA34 (uptrend) or < EMA34 (downtrend)
        uptrend = close_val > ema_34_1d_val
        downtrend = close_val < ema_34_1d_val
        
        if position == 0:
            # Long: break above R1 with uptrend and volume spike
            # Short: break below S1 with downtrend and volume spike
            long_signal = (high_val > r1_val and uptrend and vol_spike)
            short_signal = (low_val < s1_val and downtrend and vol_spike)
            
            if long_signal:
                signals[i] = 0.25
                position = 1
                entry_price = close_val
                max_high_since_entry = high_val
                min_low_since_entry = low_val
            elif short_signal:
                signals[i] = -0.25
                position = -1
                entry_price = close_val
                max_high_since_entry = high_val
                min_low_since_entry = low_val
            else:
                signals[i] = 0.0
        elif position == 1:
            # Hold long
            signals[i] = 0.25
            max_high_since_entry = max(max_high_since_entry, high_val)
            min_low_since_entry = min(min_low_since_entry, low_val)
            
            # Exit: ATR-based stoploss or trend reversal
            if low_val <= entry_price - 2.0 * atr_val or close_val < ema_34_1d_val:
                signals[i] = 0.0
                position = 0
        elif position == -1:
            # Hold short
            signals[i] = -0.25
            max_high_since_entry = max(max_high_since_entry, high_val)
            min_low_since_entry = min(min_low_since_entry, low_val)
            
            # Exit: ATR-based stoploss or trend reversal
            if high_val >= entry_price + 2.0 * atr_val or close_val > ema_34_1d_val:
                signals[i] = 0.0
                position = 0
    
    return signals

name = "4h_Camarilla_R1S1_Breakout_1dEMA34_Trend_ATRStop_v2"
timeframe = "4h"
leverage = 1.0
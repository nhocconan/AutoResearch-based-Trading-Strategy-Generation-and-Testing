# 1H_Pivot_R1_S1_Breakout_Volume_Session
# Hypothesis: Use daily Camarilla pivot levels (R1/S1) for direction on 1H timeframe.
# Only trade during 08-20 UTC session to reduce noise. Enter on break of R1 (long) or S1 (short)
# with volume confirmation (>1.5x average). Exit on opposite pivot touch.
# Designed for low-frequency, high-conviction trades to avoid fee drag.
# Target: 15-35 trades/year per symbol.

#!/usr/bin/env python3
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
    
    # === Daily data for Camarilla pivots ===
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Camarilla pivot calculations (based on previous day)
    # R1 = close + 1.1*(high-low)/12
    # S1 = close - 1.1*(high-low)/12
    prev_high = np.roll(high_1d, 1)
    prev_low = np.roll(low_1d, 1)
    prev_close = np.roll(close_1d, 1)
    prev_high[0] = np.nan
    prev_low[0] = np.nan
    prev_close[0] = np.nan
    
    camarilla_r1 = prev_close + 1.1 * (prev_high - prev_low) / 12
    camarilla_s1 = prev_close - 1.1 * (prev_high - prev_low) / 12
    
    # Align to 1H timeframe (only use after daily candle closes)
    r1_aligned = align_htf_to_ltf(prices, df_1d, camarilla_r1)
    s1_aligned = align_htf_to_ltf(prices, df_1d, camarilla_s1)
    
    # === Volume confirmation ===
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_ratio = volume / vol_ma_20
    
    # === Session filter: 08-20 UTC ===
    hours = prices.index.hour
    
    signals = np.zeros(n)
    
    # Warmup
    warmup = 50
    
    # Track position
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(warmup, n):
        # Skip if any data is NaN
        if (np.isnan(r1_aligned[i]) or np.isnan(s1_aligned[i]) or 
            np.isnan(vol_ratio[i])):
            signals[i] = 0.0
            position = 0
            continue
        
        hour = hours[i]
        in_session = (8 <= hour <= 20)
        
        price = close[i]
        r1 = r1_aligned[i]
        s1 = s1_aligned[i]
        vol_ratio_val = vol_ratio[i]
        
        # Exit conditions
        if position == 1:  # Long
            if price <= s1:  # Exit at S1
                signals[i] = 0.0
                position = 0
                continue
        elif position == -1:  # Short
            if price >= r1:  # Exit at R1
                signals[i] = 0.0
                position = 0
                continue
        
        # Entry conditions (only when flat)
        if position == 0 and in_session:
            # Long: Break above R1 with volume
            if price > r1 and vol_ratio_val > 1.5:
                signals[i] = 0.20
                position = 1
                continue
            # Short: Break below S1 with volume
            elif price < s1 and vol_ratio_val > 1.5:
                signals[i] = -0.20
                position = -1
                continue
        
        # Hold position
        if position == 1:
            signals[i] = 0.20
        elif position == -1:
            signals[i] = -0.20
        else:
            signals[i] = 0.0
    
    return signals

name = "1H_Pivot_R1_S1_Breakout_Volume_Session"
timeframe = "1h"
leverage = 1.0
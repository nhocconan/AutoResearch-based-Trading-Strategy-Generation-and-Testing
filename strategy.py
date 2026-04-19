#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4-hour Camarilla R1/S1 breakout with daily EMA34 filter and volume spike confirmation.
# Long when: Price breaks above R1, daily EMA34 upward, volume > 1.5x 20-period average
# Short when: Price breaks below S1, daily EMA34 downward, volume > 1.5x 20-period average
# Exit when: Price crosses back through the pivot point (PP)
# Camarilla levels provide institutional support/resistance, EMA34 filters trend, volume confirms breakout strength.
# Target: 20-30 trades/year per symbol. Works in bull (buy breakouts) and bear (sell breakdowns).
name = "4h_Camarilla_R1_S1_Breakout_Volume_EMA34Filter"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # 1-day data for Camarilla pivot levels and EMA34
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate daily pivot point (PP) and Camarilla levels
    # PP = (H + L + C) / 3
    pp_1d = (high_1d + low_1d + close_1d) / 3.0
    # R1 = C + (H - L) * 1.1 / 12
    r1_1d = close_1d + (high_1d - low_1d) * 1.1 / 12.0
    # S1 = C - (H - L) * 1.1 / 12
    s1_1d = close_1d - (high_1d - low_1d) * 1.1 / 12.0
    
    # Calculate EMA34 on daily data for trend filter
    ema34_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    
    # Align 1D data to 4H timeframe
    pp_1d_aligned = align_htf_to_ltf(prices, df_1d, pp_1d)
    r1_1d_aligned = align_htf_to_ltf(prices, df_1d, r1_1d)
    s1_1d_aligned = align_htf_to_ltf(prices, df_1d, s1_1d)
    ema34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema34_1d)
    
    # 20-period volume average for confirmation
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 34  # Wait for EMA34 calculation
    
    for i in range(start_idx, n):
        # Skip if any required data is not available
        if (np.isnan(pp_1d_aligned[i]) or np.isnan(r1_1d_aligned[i]) or 
            np.isnan(s1_1d_aligned[i]) or np.isnan(ema34_1d_aligned[i]) or 
            np.isnan(vol_ma_20[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        pp = pp_1d_aligned[i]
        r1 = r1_1d_aligned[i]
        s1 = s1_1d_aligned[i]
        ema34 = ema34_1d_aligned[i]
        vol = volume[i]
        vol_ma = vol_ma_20[i]
        
        if position == 0:
            # Long entry: Price breaks above R1, EMA34 upward, volume spike
            if (price > r1 and close[i-1] <= r1 and 
                ema34 > ema34_1d_aligned[i-1] and vol > 1.5 * vol_ma):
                signals[i] = 0.25
                position = 1
            # Short entry: Price breaks below S1, EMA34 downward, volume spike
            elif (price < s1 and close[i-1] >= s1 and 
                  ema34 < ema34_1d_aligned[i-1] and vol > 1.5 * vol_ma):
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Long exit: Price crosses back below pivot point
            if price < pp:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: Price crosses back above pivot point
            if price > pp:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals
#!/usr/bin/env python3
# 12h_camarilla_pivot_1d_volume_v1
# Hypothesis: 12h Camarilla pivot levels (H3/L3) from 1d HTF act as support/resistance with volume confirmation.
# In bull markets: buy near L3 with volume spike; in bear markets: sell near H3 with volume spike.
# Uses 1d HTF for pivot calculation (properly aligned) and 12h EMA(50) for trend filter.
# Target: 12-30 trades/year (50-120 total over 4 years).

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "12h_camarilla_pivot_1d_volume_v1"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # 1d HTF data for Camarilla pivots
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    # Calculate Camarilla levels from previous 1d bar
    # H3 = close + 1.1*(high-low)/2, L3 = close - 1.1*(high-low)/2
    prev_close = df_1d['close'].shift(1).values
    prev_high = df_1d['high'].shift(1).values
    prev_low = df_1d['low'].shift(1).values
    camarilla_h3 = prev_close + 1.1 * (prev_high - prev_low) / 2
    camarilla_l3 = prev_close - 1.1 * (prev_high - prev_low) / 2
    
    # Align Camarilla levels to 12h timeframe
    camarilla_h3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_h3)
    camarilla_l3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_l3)
    
    # 12h EMA(50) for trend filter
    ema_50 = pd.Series(close).ewm(span=50, min_periods=50, adjust=False).mean().values
    
    # Volume confirmation: current volume > 2.0x 20-period average
    volume_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(50, n):
        # Skip if any required data is NaN
        if (np.isnan(camarilla_h3_aligned[i]) or np.isnan(camarilla_l3_aligned[i]) or
            np.isnan(ema_50[i]) or np.isnan(volume_ma[i])):
            signals[i] = 0.0
            continue
        
        if position == 1:  # Long position
            # Exit: price crosses below L3 OR trend turns bearish
            if close[i] < camarilla_l3_aligned[i] or close[i] < ema_50[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit: price crosses above H3 OR trend turns bullish
            if close[i] > camarilla_h3_aligned[i] or close[i] > ema_50[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat
            # Volume confirmation
            volume_confirmed = volume[i] > 2.0 * volume_ma[i]
            
            if volume_confirmed:
                # Long setup: price near L3 and above EMA50 (bullish alignment)
                if close[i] <= camarilla_l3_aligned[i] * 1.005 and close[i] > ema_50[i]:
                    position = 1
                    signals[i] = 0.25
                # Short setup: price near H3 and below EMA50 (bearish alignment)
                elif close[i] >= camarilla_h3_aligned[i] * 0.995 and close[i] < ema_50[i]:
                    position = -1
                    signals[i] = -0.25
    
    return signals
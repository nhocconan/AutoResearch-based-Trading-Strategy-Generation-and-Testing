#!/usr/bin/env python3
name = "1d_Camarilla_Pivot_Support_Resistance_Bounce"
timeframe = "1d"
leverage = 1.0

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
    
    # Calculate 1w EMA8 for trend filter (HTF)
    df_1w = get_htf_data(prices, '1w')
    ema8_1w = pd.Series(df_1w['close']).ewm(span=8, min_periods=8, adjust=False).mean().values
    ema8_1w_aligned = align_htf_to_ltf(prices, df_1w, ema8_1w)
    
    # Calculate daily Camarilla pivot levels (based on previous day)
    prev_close = np.roll(close, 1)
    prev_high = np.roll(high, 1)
    prev_low = np.roll(low, 1)
    prev_close[0] = close[0]  # avoid NaN for first bar
    prev_high[0] = high[0]
    prev_low[0] = low[0]
    
    # Camarilla levels (based on previous day's range)
    camarilla_range = prev_high - prev_low
    camarilla_r4 = prev_close + camarilla_range * 1.1 / 2
    camarilla_r3 = prev_close + camarilla_range * 1.1 / 4
    camarilla_s3 = prev_close - camarilla_range * 1.1 / 4
    camarilla_s4 = prev_close - camarilla_range * 1.1 / 2
    
    # Volume filter: 20-period EMA
    vol_ema20 = pd.Series(volume).ewm(span=20, min_periods=20, adjust=False).mean().values
    volume_ok = volume > vol_ema20 * 1.5
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after warmup
    start_idx = 40
    
    for i in range(start_idx, n):
        # Skip if any required data is invalid
        if (np.isnan(ema8_1w_aligned[i]) or np.isnan(camarilla_r4[i]) or 
            np.isnan(camarilla_r3[i]) or np.isnan(camarilla_s3[i]) or 
            np.isnan(camarilla_s4[i]) or np.isnan(volume_ok[i])):
            if position == 1:
                signals[i] = 0.25
            elif position == -1:
                signals[i] = -0.25
            else:
                signals[i] = 0.0
            continue
        
        # Conditions
        price_above_ema1w = close[i] > ema8_1w_aligned[i]
        price_below_ema1w = close[i] < ema8_1w_aligned[i]
        near_support = (close[i] <= camarilla_s3[i] * 1.01) and (close[i] >= camarilla_s4[i] * 0.99)
        near_resistance = (close[i] <= camarilla_r4[i] * 1.01) and (close[i] >= camarilla_r3[i] * 0.99)
        
        if position == 0:
            # Long: Price near S3/S4 support + above weekly EMA8 + volume spike
            if near_support and price_above_ema1w and volume_ok[i]:
                signals[i] = 0.25
                position = 1
            # Short: Price near R3/R4 resistance + below weekly EMA8 + volume spike
            elif near_resistance and price_below_ema1w and volume_ok[i]:
                signals[i] = -0.25
                position = -1
        else:
            # Exit conditions
            if position == 1:
                # Exit: Price crosses above R3 or trend reverses
                if close[i] > camarilla_r3[i] or close[i] < ema8_1w_aligned[i]:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = 0.25
            elif position == -1:
                # Exit: Price crosses below S3 or trend reverses
                if close[i] < camarilla_s3[i] or close[i] > ema8_1w_aligned[i]:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = -0.25
    
    return signals
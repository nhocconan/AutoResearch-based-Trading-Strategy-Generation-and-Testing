#!/usr/bin/env python3
"""
12h Camarilla Pivot + 1d Trend + Volume Confirmation
Hypothesis: Camarilla pivot levels provide high-probability reversal zones in ranging markets. Combined with 1d trend filter and volume confirmation to capture reversals with momentum. Works in bull/bear by using volatility-adjusted pivots and trend alignment. Targets 15-35 trades/year on 12h timeframe.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "12h_camarilla_pivot_1d_trend_volume_v1"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Price data
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # 1d EMA(50) for trend filter
    df_1d = get_htf_data(prices, '1d')
    ema_50_1d = df_1d['close'].ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # 1d ATR(14) for volatility filter
    tr_1d = np.maximum(df_1d['high'] - df_1d['low'], 
                       np.maximum(np.abs(df_1d['high'] - np.roll(df_1d['close'], 1)), 
                                  np.abs(df_1d['low'] - np.roll(df_1d['close'], 1))))
    tr_1d[0] = df_1d['high'].iloc[0] - df_1d['low'].iloc[0]
    atr_1d = pd.Series(tr_1d).rolling(window=14, min_periods=14).mean().values
    atr_1d_aligned = align_htf_to_ltf(prices, df_1d, atr_1d)
    
    # Previous 12h bar data for Camarilla calculation
    prev_high = np.roll(high, 1)
    prev_low = np.roll(low, 1)
    prev_close = np.roll(close, 1)
    prev_high[0] = high[0]
    prev_low[0] = low[0]
    prev_close[0] = close[0]
    
    # Camarilla levels (based on previous 12h bar)
    range_val = prev_high - prev_low
    camarilla_h4 = prev_close + (range_val * 1.1 / 2)
    camarilla_l4 = prev_close - (range_val * 1.1 / 2)
    camarilla_h3 = prev_close + (range_val * 1.1 / 4)
    camarilla_l3 = prev_close - (range_val * 1.1 / 4)
    
    # Volume filter (>1.3x 20-period average)
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_filter = volume > (vol_ma * 1.3)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(100, n):
        # Skip if any required data is NaN
        if (np.isnan(ema_50_1d_aligned[i]) or np.isnan(atr_1d_aligned[i]) or 
            np.isnan(camarilla_h4) or np.isnan(camarilla_l4) or
            np.isnan(camarilla_h3) or np.isnan(camarilla_l3) or
            np.isnan(vol_filter[i])):
            signals[i] = 0.0
            continue
        
        if position == 1:  # Long position
            # Exit: price reaches L3 (take profit) OR trend reverses OR volatility spike
            if (close[i] >= camarilla_l3[i] or 
                close[i] < ema_50_1d_aligned[i] or
                atr_1d_aligned[i] > (atr_1d_aligned[i-20] * 2.0 if i >= 20 else atr_1d_aligned[i])):
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit: price reaches H3 (take profit) OR trend reverses OR volatility spike
            if (close[i] <= camarilla_h3[i] or 
                close[i] > ema_50_1d_aligned[i] or
                atr_1d_aligned[i] > (atr_1d_aligned[i-20] * 2.0 if i >= 20 else atr_1d_aligned[i])):
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat, look for entry
            # Long at L4 with trend alignment and volume
            if (close[i] <= camarilla_l4[i] and 
                close[i] > ema_50_1d_aligned[i] and 
                vol_filter[i]):
                position = 1
                signals[i] = 0.25
            # Short at H4 with trend alignment and volume
            elif (close[i] >= camarilla_h4[i] and 
                  close[i] < ema_50_1d_aligned[i] and 
                  vol_filter[i]):
                position = -1
                signals[i] = -0.25
    
    return signals
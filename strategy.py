#!/usr/bin/env python3
"""
Hypothesis: 12h strategy using Camarilla pivot levels (S3/S4 and R3/R4) from daily timeframe 
combined with 12h EMA trend filter and volume confirmation. 
In bull markets: buy near S3/S4 support in uptrend with volume confirmation.
In bear markets: sell near R3/R4 resistance in downtrend with volume confirmation.
Camarilla levels provide institutional support/resistance that work in both trending and ranging markets.
Target: 15-25 trades/year to minimize fee drag while capturing significant reversals at key levels.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Load daily data ONCE before loop for Camarilla pivots and EMA trend
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    # Calculate Camarilla pivot levels from previous day
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Camarilla levels: based on previous day's range
    range_1d = high_1d - low_1d
    # S1 = C - (H-L)*1.0833/2, S2 = C - (H-L)*1.1666/2, S3 = C - (H-L)*1.2500/2, S4 = C - (H-L)*1.5000/2
    # R1 = C + (H-L)*1.0833/2, R2 = C + (H-L)*1.1666/2, R3 = C + (H-L)*1.2500/2, R4 = C + (H-L)*1.5000/2
    camarilla_s3 = close_1d - range_1d * 1.2500 / 2
    camarilla_s4 = close_1d - range_1d * 1.5000 / 2
    camarilla_r3 = close_1d + range_1d * 1.2500 / 2
    camarilla_r4 = close_1d + range_1d * 1.5000 / 2
    
    # Align Camarilla levels to 12h timeframe (need previous day's levels available)
    camarilla_s3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_s3)
    camarilla_s4_aligned = align_htf_to_ltf(prices, df_1d, camarilla_s4)
    camarilla_r3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_r3)
    camarilla_r4_aligned = align_htf_to_ltf(prices, df_1d, camarilla_r4)
    
    # 12h EMA34 for trend filter
    close_12h = prices['close'].values
    ema_34 = pd.Series(close_12h).ewm(span=34, adjust=False, min_periods=34).mean().values
    
    # Volume confirmation: volume > 1.8x 20-period average
    vol_ma_20 = pd.Series(prices['volume'].values).rolling(window=20, min_periods=20).mean().values
    vol_ratio = prices['volume'].values / vol_ma_20
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(34, n):  # Start after EMA warmup
        # Skip if indicators not ready
        if (np.isnan(camarilla_s3_aligned[i]) or np.isnan(camarilla_s4_aligned[i]) or 
            np.isnan(camarilla_r3_aligned[i]) or np.isnan(camarilla_r4_aligned[i]) or
            np.isnan(ema_34[i]) or np.isnan(vol_ratio[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        price_close = prices['close'].iloc[i]
        price_low = prices['low'].iloc[i]
        price_high = prices['high'].iloc[i]
        ema_trend = ema_34[i]
        vol_ratio_val = vol_ratio[i]
        vol_threshold = 1.8  # Volume spike filter
        
        if position == 0:
            # Enter long: price touches S3/S4 support + uptrend (price > EMA34) + volume spike
            support_touched = (price_low <= camarilla_s3_aligned[i] or price_low <= camarilla_s4_aligned[i])
            if (support_touched and 
                price_close > ema_trend and 
                vol_ratio_val > vol_threshold):
                signals[i] = 0.25
                position = 1
            # Enter short: price touches R3/R4 resistance + downtrend (price < EMA34) + volume spike
            elif (price_high >= camarilla_r3_aligned[i] or price_high >= camarilla_r4_aligned[i]) and \
                 price_close < ema_trend and \
                 vol_ratio_val > vol_threshold:
                signals[i] = -0.25
                position = -1
        
        elif position != 0:
            # Exit: trend reversal or price reaches opposite Camarilla level
            if position == 1:
                # Exit long: trend reversal or price reaches R3/R4
                if price_close < ema_34[i] or price_high >= camarilla_r3_aligned[i] or price_high >= camarilla_r4_aligned[i]:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = 0.25
            else:  # position == -1
                # Exit short: trend reversal or price reaches S3/S4
                if price_close > ema_34[i] or price_low <= camarilla_s3_aligned[i] or price_low <= camarilla_s4_aligned[i]:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = -0.25
    
    return signals

name = "12h_Camarilla_S3S4_R3R4_EMA34_Volume"
timeframe = "12h"
leverage = 1.0
#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Camarilla R3/S3 breakout with 1d EMA trend filter and volume confirmation.
# Uses daily Camarilla pivot levels for structural support/resistance, requiring price to break
# through R3 (short) or S3 (long) with volume confirmation and alignment to 1d EMA trend.
# Works in bull markets via R3 breakouts and in bear markets via S3 breakdowns.
# Volume ensures conviction, EMA filter avoids counter-trend trades.
name = "6h_Camarilla_R3S3_Breakout_1dEMA_Trend_Volume"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1d data for Camarilla pivots and EMA
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    # Calculate Camarilla levels from previous 1d bar
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Camarilla levels (based on previous day)
    range_1d = high_1d - low_1d
    camarilla_r3 = close_1d + (range_1d * 1.1 / 4)
    camarilla_s3 = close_1d - (range_1d * 1.1 / 4)
    camarilla_r4 = close_1d + (range_1d * 1.1 / 2)
    camarilla_s4 = close_1d - (range_1d * 1.1 / 2)
    
    # Align Camarilla levels to 6t (wait for previous day's close)
    camarilla_r3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_r3)
    camarilla_s3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_s3)
    camarilla_r4_aligned = align_htf_to_ltf(prices, df_1d, camarilla_r4)
    camarilla_s4_aligned = align_htf_to_ltf(prices, df_1d, camarilla_s4)
    
    # 1d EMA34 for trend filter
    ema_34_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # Volume confirmation: volume > 1.5x 20-period EMA
    vol_ema20 = pd.Series(volume).ewm(span=20, adjust=False, min_periods=20).mean().values
    vol_confirm = volume > (1.5 * vol_ema20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 34  # Ensure enough data for EMA
    
    for i in range(start_idx, n):
        # Skip if required data unavailable
        if (np.isnan(ema_34_aligned[i]) or np.isnan(camarilla_r3_aligned[i]) or 
            np.isnan(camarilla_s3_aligned[i]) or np.isnan(vol_ema20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        price = close[i]
        
        if position == 0:
            # Long: price breaks above S3 (support break = bullish in uptrend) 
            # Wait for breakdown of S3 as continuation of downtrend? No - S3 is support.
            # Actually: In Camarilla, S3/S4 are strong support/resistance.
            # Break below S3 = bearish, break above R3 = bullish.
            # But for breakout strategy: break above R3 = long, break below S3 = short.
            if price > camarilla_r3_aligned[i] and vol_confirm[i] and price > ema_34_aligned[i]:
                signals[i] = 0.25
                position = 1
            elif price < camarilla_s3_aligned[i] and vol_confirm[i] and price < ema_34_aligned[i]:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit long: price crosses back below R3 or reaches R4 (take profit)
            if price < camarilla_r3_aligned[i] or price > camarilla_r4_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: price crosses back above S3 or reaches S4 (take profit)
            if price > camarilla_s3_aligned[i] or price < camarilla_s4_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals
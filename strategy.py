#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Camarilla pivot reversal strategy with 1d trend filter and volume confirmation
# Uses Camarilla pivot levels from daily timeframe to identify reversal zones: 
# - Fade at R3/S3 (sell at resistance, buy at support) in ranging markets
# - Breakout continuation at R4/S4 (buy break above R4, sell break below S4) in trending markets
# - 1d EMA50 filter ensures trades align with higher timeframe trend
# - Volume confirmation avoids false breakouts
# Designed for low frequency (target: 15-30 trades/year) to minimize fee impact
# Works in both bull/bear via adaptive logic: mean reversion in range, trend following in breakout

name = "6h_camarilla_pivot_1d_ema_volume_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Price data
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # 1d Camarilla pivot levels
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate Camarilla levels for each day
    pivot = (high_1d + low_1d + close_1d) / 3
    range_1d = high_1d - low_1d
    
    # Resistance levels
    r1 = pivot + (range_1d * 1.0833 / 12)
    r2 = pivot + (range_1d * 1.1666 / 6)
    r3 = pivot + (range_1d * 1.2500 / 4)
    r4 = pivot + (range_1d * 1.5000 / 2)
    
    # Support levels
    s1 = pivot - (range_1d * 1.0833 / 12)
    s2 = pivot - (range_1d * 1.1666 / 6)
    s3 = pivot - (range_1d * 1.2500 / 4)
    s4 = pivot - (range_1d * 1.5000 / 2)
    
    # Align Camarilla levels to 6h timeframe
    pivot_aligned = align_htf_to_ltf(prices, df_1d, pivot)
    r3_aligned = align_htf_to_ltf(prices, df_1d, r3)
    r4_aligned = align_htf_to_ltf(prices, df_1d, r4)
    s3_aligned = align_htf_to_ltf(prices, df_1d, s3)
    s4_aligned = align_htf_to_ltf(prices, df_1d, s4)
    
    # 1d trend filter (EMA50)
    ema_50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # Volume confirmation (20-period average)
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # Track position: 1=long, -1=short, 0=flat
    
    for i in range(50, n):
        # Skip if required data not available
        if (np.isnan(pivot_aligned[i]) or np.isnan(r3_aligned[i]) or np.isnan(r4_aligned[i]) or
            np.isnan(s3_aligned[i]) or np.isnan(s4_aligned[i]) or np.isnan(ema_50_1d_aligned[i]) or
            np.isnan(vol_ma[i])):
            signals[i] = 0.0
            continue
        
        # Volume confirmation: current volume above average
        vol_confirm = volume[i] > vol_ma[i]
        
        # Market regime: trending if price outside R3/S3, ranging if inside
        price_above_r3 = close[i] > r3_aligned[i]
        price_below_s3 = close[i] < s3_aligned[i]
        price_inside = (close[i] >= s3_aligned[i]) & (close[i] <= r3_aligned[i])
        
        # Trend filter from 1d EMA
        uptrend = close[i] > ema_50_1d_aligned[i]
        downtrend = close[i] < ema_50_1d_aligned[i]
        
        # Exit conditions
        if position == 1:  # Long position
            # Exit on reverse signal or at opposite pivot level
            if (close[i] < s3_aligned[i]) or (close[i] > r4_aligned[i] and not uptrend):
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25  # Maintain long position
        elif position == -1:  # Short position
            # Exit on reverse signal or at opposite pivot level
            if (close[i] > r3_aligned[i]) or (close[i] < s4_aligned[i] and not downtrend):
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25  # Maintain short position
        else:  # Flat, look for entry
            # Ranging market: mean reversion at S3/R3
            if price_inside:
                # Buy at S3 with uptrend and volume
                if (close[i] <= s3_aligned[i] * 1.001) and uptrend and vol_confirm:  # Small buffer for entry
                    position = 1
                    signals[i] = 0.25
                # Sell at R3 with downtrend and volume
                elif (close[i] >= r3_aligned[i] * 0.999) and downtrend and vol_confirm:
                    position = -1
                    signals[i] = -0.25
            # Trending market: breakout continuation at S4/R4
            else:
                # Buy breakout above R4 with uptrend and volume
                if price_above_r3 and (close[i] > r4_aligned[i] * 1.001) and uptrend and vol_confirm:
                    position = 1
                    signals[i] = 0.25
                # Sell breakdown below S4 with downtrend and volume
                elif price_below_s3 and (close[i] < s4_aligned[i] * 0.999) and downtrend and vol_confirm:
                    position = -1
                    signals[i] = -0.25
    
    return signals
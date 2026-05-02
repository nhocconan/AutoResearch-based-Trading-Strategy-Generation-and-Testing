#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Williams Alligator + Elder Ray combination with 1w trend filter
# Uses 6h timeframe for signal generation with Williams Alligator (jaw/teeth/lips) for trend direction
# Elder Ray (Bull/Bear power) measures buying/selling pressure relative to 13-period EMA
# 1-week EMA50 provides higher timeframe trend filter to avoid counter-trend trades
# Discrete position sizing (0.25) balances profit potential with fee drag minimization
# Target: 50-150 total trades over 4 years (12-37/year) for 6h timeframe
# Works in both bull and bear markets by only trading in direction of 1w trend and requiring
# confluence of Alligator alignment and Elder Ray strength

name = "6h_WilliamsAlligator_ElderRay_1wEMA50_Trend_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Get 1w data for trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 50:
        return np.zeros(n)
    
    # 1w EMA50 trend filter
    ema50_1w = pd.Series(df_1w['close']).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema50_1w)
    
    # Williams Alligator on 6h (jaw=13, teeth=8, lips=5)
    # Jaw: 13-period SMMA (smoothed moving average) of median price
    # Teeth: 8-period SMMA of median price
    # Lips: 5-period SMMA of median price
    median_price = (high + low) / 2.0
    
    # Smoothed Moving Average (SMMA) approximation using EMA with alpha=1/period
    def smma(data, period):
        if len(data) < period:
            return np.full_like(data, np.nan)
        result = np.full_like(data, np.nan)
        alpha = 1.0 / period
        result[period-1] = np.mean(data[:period])  # Initialize with SMA
        for i in range(period, len(data)):
            result[i] = alpha * data[i] + (1 - alpha) * result[i-1]
        return result
    
    jaw = smma(median_price, 13)
    teeth = smma(median_price, 8)
    lips = smma(median_price, 5)
    
    # Elder Ray: Bull Power = High - EMA13, Bear Power = Low - EMA13
    ema13 = pd.Series(close).ewm(span=13, adjust=False, min_periods=13).mean().values
    bull_power = high - ema13
    bear_power = low - ema13
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after warmup
    start_idx = 50
    
    for i in range(start_idx, n):
        # Check for NaN values in indicators
        if (np.isnan(ema50_1w_aligned[i]) or np.isnan(jaw[i]) or np.isnan(teeth[i]) or 
            np.isnan(lips[i]) or np.isnan(bull_power[i]) or np.isnan(bear_power[i])):
            signals[i] = 0.0
            continue
        
        if position == 0:  # Flat - look for new entries
            # Alligator aligned for uptrend: Lips > Teeth > Jaw
            # Elder Ray confirmation: Bull Power > 0 (buying pressure)
            # 1w trend filter: price > 1w EMA50
            if (lips[i] > teeth[i] > jaw[i] and 
                bull_power[i] > 0 and 
                close[i] > ema50_1w_aligned[i]):
                signals[i] = 0.25
                position = 1
            # Alligator aligned for downtrend: Jaw > Teeth > Lips
            # Elder Ray confirmation: Bear Power < 0 (selling pressure)
            # 1w trend filter: price < 1w EMA50
            elif (jaw[i] > teeth[i] > lips[i] and 
                  bear_power[i] < 0 and 
                  close[i] < ema50_1w_aligned[i]):
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        
        elif position == 1:  # Long position
            # Exit: Alligator reverses (Teeth < Jaw) or Elder Ray weakens
            # or 1w trend fails
            if (teeth[i] < jaw[i] or 
                bull_power[i] <= 0 or 
                close[i] < ema50_1w_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:  # Short position
            # Exit: Alligator reverses (Teeth > Lips) or Elder Ray weakens
            # or 1w trend fails
            if (teeth[i] > lips[i] or 
                bear_power[i] >= 0 or 
                close[i] > ema50_1w_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals
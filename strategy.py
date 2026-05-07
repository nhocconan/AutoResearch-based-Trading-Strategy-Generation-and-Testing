#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Williams Alligator with 1d trend filter and volume confirmation.
# Long when Alligator jaw < teeth < lips (bullish alignment) AND price > 1d EMA50 with volume spike.
# Short when Alligator jaw > teeth > lips (bearish alignment) AND price < 1d EMA50 with volume spike.
# Uses Williams Alligator (SMMA-based) to identify trend alignment and avoid choppy markets.
# 1d EMA50 filter ensures alignment with higher timeframe trend. Volume spike confirms momentum.
# Designed for ~20-30 trades/year to minimize fee drag while capturing strong trends.
name = "4h_WilliamsAlligator_1dEMA50_Volume"
timeframe = "4h"
leverage = 1.0

def smma(data, period):
    """Smoothed Moving Average (SMMA)"""
    if len(data) < period:
        return np.full_like(data, np.nan, dtype=float)
    result = np.full_like(data, np.nan, dtype=float)
    # First value is SMA
    result[period-1] = np.mean(data[:period])
    # Subsequent values: SMMA = (prev_smma * (period-1) + current_price) / period
    for i in range(period, len(data)):
        result[i] = (result[i-1] * (period-1) + data[i]) / period
    return result

def generate_signals(prices):
    n = len(prices)
    if n < 60:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Load 1d data ONCE for trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 60:
        return np.zeros(n)
    
    # 1d trend filter: 50-period EMA on close
    ema_50_1d = pd.Series(df_1d['close']).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # Williams Alligator on 4h data: SMMA(13,8), SMMA(8,5), SMMA(5,3)
    # Jaw: SMMA(13,8), Teeth: SMMA(8,5), Lips: SMMA(5,3)
    jaw = smma(high, 13)  # Using high for jaw as per typical Alligator
    teeth = smma(high, 8)  # Using high for teeth
    lips = smma(high, 5)   # Using high for lips
    
    # Volume spike detection: current volume > 2.0 * 20-period EMA of volume
    vol_ema_20 = pd.Series(volume).ewm(span=20, adjust=False, min_periods=20).mean().values
    vol_spike = np.where(vol_ema_20 > 0, volume / vol_ema_20, 1.0) > 2.0
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 60  # Sufficient warmup for Alligator calculations
    
    for i in range(start_idx, n):
        if (np.isnan(ema_50_1d_aligned[i]) or np.isnan(jaw[i]) or 
            np.isnan(teeth[i]) or np.isnan(lips[i]) or np.isnan(vol_spike[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Alligator alignment: jaw < teeth < lips = bullish, jaw > teeth > lips = bearish
        bullish_alignment = (jaw[i] < teeth[i]) and (teeth[i] < lips[i])
        bearish_alignment = (jaw[i] > teeth[i]) and (teeth[i] > lips[i])
        
        # Trend filter: price above/below 1d EMA50
        uptrend = close[i] > ema_50_1d_aligned[i]
        downtrend = close[i] < ema_50_1d_aligned[i]
        
        if position == 0:
            # Long condition: bullish Alligator alignment, uptrend, volume spike
            long_condition = bullish_alignment and uptrend and vol_spike[i]
            # Short condition: bearish Alligator alignment, downtrend, volume spike
            short_condition = bearish_alignment and downtrend and vol_spike[i]
            
            if long_condition:
                signals[i] = 0.25
                position = 1
            elif short_condition:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit: Alligator turns bearish OR trend turns down
            if (not bullish_alignment) or (not uptrend):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit: Alligator turns bullish OR trend turns up
            if (not bearish_alignment) or (not downtrend):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals
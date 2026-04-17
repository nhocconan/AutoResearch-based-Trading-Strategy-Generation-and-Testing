#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Williams Alligator + Elder Ray combination with 1d trend filter.
# Uses Alligator (Jaw/Teeth/Lips) for trend direction and Elder Ray (Bull/Bear Power) for momentum.
# 1d EMA50 filter ensures alignment with higher timeframe trend.
# Designed to work in bull (trend following with Alligator alignment) and bear (mean reversion via Elder Ray extremes).
# Target: 15-30 trades/year to avoid fee drag on 6h timeframe.

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get daily data for EMA filter
    df_1d = get_htf_data(prices, '1d')
    close_1d = df_1d['close'].values
    
    # Calculate daily EMA50 for trend filter
    close_1d_series = pd.Series(close_1d)
    ema50_1d = close_1d_series.ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Align daily EMA50 to 6h
    ema50_6h = align_htf_to_ltf(prices, df_1d, ema50_1d)
    
    # Williams Alligator: SMAs of median price (HL/2)
    median_price = (high + low) / 2
    jaw = pd.Series(median_price).rolling(window=13, min_periods=13).mean().shift(8).values  # 13-period, 8-shift
    teeth = pd.Series(median_price).rolling(window=8, min_periods=8).mean().shift(5).values   # 8-period, 5-shift
    lips = pd.Series(median_price).rolling(window=5, min_periods=5).mean().shift(3).values    # 5-period, 3-shift
    
    # Elder Ray: Bull Power = High - EMA13, Bear Power = Low - EMA13
    ema13 = pd.Series(close).ewm(span=13, adjust=False, min_periods=13).mean().values
    bull_power = high - ema13
    bear_power = low - ema13
    
    signals = np.zeros(n)
    position = 0  # -1: short, 0: flat, 1: long
    
    start_idx = 50  # Need daily EMA50 and Alligator components
    
    for i in range(start_idx, n):
        # Skip if any required data is not available
        if (np.isnan(ema50_6h[i]) or 
            np.isnan(jaw[i]) or 
            np.isnan(teeth[i]) or 
            np.isnan(lips[i]) or
            np.isnan(bull_power[i]) or
            np.isnan(bear_power[i])):
            signals[i] = 0.0
            continue
        
        # Alligator alignment: Lips > Teeth > Jaw = uptrend, Lips < Teeth < Jaw = downtrend
        alligator_long = lips[i] > teeth[i] and teeth[i] > jaw[i]
        alligator_short = lips[i] < teeth[i] and teeth[i] < jaw[i]
        
        # Elder Ray strength
        strong_bull = bull_power[i] > 0 and bull_power[i] > np.abs(bear_power[i])
        strong_bear = bear_power[i] < 0 and np.abs(bear_power[i]) > bull_power[i]
        
        # Trend filter: price relative to daily EMA50
        price_above_ema = close[i] > ema50_6h[i]
        price_below_ema = close[i] < ema50_6h[i]
        
        if position == 0:
            # Long: Alligator aligned up + Elder Ray bullish + price above daily EMA
            if (alligator_long and strong_bull and price_above_ema):
                signals[i] = 0.25
                position = 1
            # Short: Alligator aligned down + Elder Ray bearish + price below daily EMA
            elif (alligator_short and strong_bear and price_below_ema):
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit long: Alligator alignment breaks OR Elder Ray turns bearish
            if not (alligator_long and strong_bull):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: Alligator alignment breaks OR Elder Ray turns bullish
            if not (alligator_short and strong_bear):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "6h_Alligator_ElderRay_EMA50"
timeframe = "6h"
leverage = 1.0
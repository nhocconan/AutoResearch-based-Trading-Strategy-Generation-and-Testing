#!/usr/bin/env python3
"""
Hypothesis: 12h Williams Alligator + Elder Ray + Volume Spike with 1w EMA50 trend filter.
- Williams Alligator: Jaw(13,8), Teeth(8,5), Lips(5,3) SMAs on median price
- Elder Ray: Bull Power = High - EMA13, Bear Power = EMA13 - Low
- Volume > 2.0x 20-period average for confirmation
- 1w EMA50 as trend filter (long only above, short only below)
- Position size: 0.25 discrete level to minimize fee churn
- Target: 12-37 trades/year on 12h timeframe (50-150 total over 4 years)
- Works in both bull/bear via trend filter + momentum confirmation
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Volume confirmation: > 2.0x 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    # 1w data for EMA50 trend filter
    df_1w = get_htf_data(prices, '1w')
    close_1w = df_1w['close'].values
    
    # 1w EMA(50)
    ema_50_1w = pd.Series(close_1w).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_50_1w)
    
    # Williams Alligator on 12h data
    # Median price = (high + low) / 2
    median_price = (high + low) / 2.0
    
    # Jaw: 13-period SMA, smoothed by 8 periods
    sma_13 = pd.Series(median_price).rolling(window=13, min_periods=13).mean().values
    jaw = pd.Series(sma_13).rolling(window=8, min_periods=8).mean().values
    
    # Teeth: 8-period SMA, smoothed by 5 periods
    sma_8 = pd.Series(median_price).rolling(window=8, min_periods=8).mean().values
    teeth = pd.Series(sma_8).rolling(window=5, min_periods=5).mean().values
    
    # Lips: 5-period SMA, smoothed by 3 periods
    sma_5 = pd.Series(median_price).rolling(window=5, min_periods=5).mean().values
    lips = pd.Series(sma_5).rolling(window=3, min_periods=3).mean().values
    
    # Elder Ray on 12h data
    # EMA13 for Elder Ray
    ema_13 = pd.Series(close).ewm(span=13, adjust=False, min_periods=13).mean().values
    bull_power = high - ema_13  # High - EMA13
    bear_power = ema_13 - low   # EMA13 - Low
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start from index where all indicators are ready
    start_idx = max(20, 13, 8, 5, 50)  # Volume MA, Alligator components, EMA
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(vol_ma[i]) or 
            np.isnan(jaw[i]) or
            np.isnan(teeth[i]) or
            np.isnan(lips[i]) or
            np.isnan(ema_50_1w_aligned[i]) or
            np.isnan(bull_power[i]) or
            np.isnan(bear_power[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Volume confirmation (> 2.0x average)
        volume_confirm = volume[i] > 2.0 * vol_ma[i]
        
        # Williams Alligator alignment (all three lines in order)
        # For uptrend: Lips > Teeth > Jaw
        # For downtrend: Jaw > Teeth > Lips
        alligator_long = lips[i] > teeth[i] and teeth[i] > jaw[i]
        alligator_short = jaw[i] > teeth[i] and teeth[i] > lips[i]
        
        # Elder Ray confirmation
        # Long: Bull Power > 0 and increasing
        # Short: Bear Power > 0 and increasing
        bull_confirm = bull_power[i] > 0 and (i == start_idx or bull_power[i] > bull_power[i-1])
        bear_confirm = bear_power[i] > 0 and (i == start_idx or bear_power[i] > bear_power[i-1])
        
        if position == 0:
            # Long: Alligator uptrend AND Elder Ray bull confirmation AND price above 1w EMA50 AND volume confirmation
            if alligator_long and bull_confirm and close[i] > ema_50_1w_aligned[i] and volume_confirm:
                signals[i] = 0.25
                position = 1
            # Short: Alligator downtrend AND Elder Ray bear confirmation AND price below 1w EMA50 AND volume confirmation
            elif alligator_short and bear_confirm and close[i] < ema_50_1w_aligned[i] and volume_confirm:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Long exit: Alligator breakdown OR Elder Ray turns negative OR price crosses below 1w EMA50
            if not alligator_long or bull_power[i] <= 0 or close[i] < ema_50_1w_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: Alligator reversal OR Elder Ray turns negative OR price crosses above 1w EMA50
            if not alligator_short or bear_power[i] <= 0 or close[i] > ema_50_1w_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "12h_WilliamsAlligator_ElderRay_VolumeSpike_1wEMA50_Trend_v1"
timeframe = "12h"
leverage = 1.0
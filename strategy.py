#!/usr/bin/env python3
"""
4h_VWAP_Bounce_BullBear_v1
Strategy: VWAP bounce with Bollinger Band filter and trend filter.
Long: Price touches VWAP from above and bounces up in uptrend.
Short: Price touches VWAP from below and bounces down in downtrend.
Designed for 4h timeframe: ~15-25 trades/year per symbol (60-100 total over 4 years).
Works in bull/bear via trend filter and VWAP mean-reversion bounce logic.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Calculate typical price and VWAP
    typical_price = (high + low + close) / 3.0
    vwap_numerator = np.cumsum(typical_price * volume)
    vwap_denominator = np.cumsum(volume)
    vwap = np.divide(vwap_numerator, vwap_denominator, 
                     out=np.full_like(vwap_numerator, np.nan), 
                     where=vwap_denominator!=0)
    
    # Bollinger Bands (20, 2) on close
    sma_20 = pd.Series(close).rolling(window=20, min_periods=20).mean().values
    std_20 = pd.Series(close).rolling(window=20, min_periods=20).std().values
    upper_bb = sma_20 + 2.0 * std_20
    lower_bb = sma_20 - 2.0 * std_20
    
    # Get daily data for trend filter
    df_1d = get_htf_data(prices, '1d')
    close_1d = df_1d['close'].values
    
    # Daily EMA50 and EMA200 for trend filter
    ema_50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_200_1d = pd.Series(close_1d).ewm(span=200, adjust=False, min_periods=200).mean().values
    
    # Align daily data to 4h timeframe
    ema_50_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    ema_200_aligned = align_htf_to_ltf(prices, df_1d, ema_200_1d)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 50  # need enough for EMA200
    
    for i in range(start_idx, n):
        # Skip if any required data is not available
        if (np.isnan(vwap[i]) or np.isnan(sma_20[i]) or np.isnan(std_20[i]) or
            np.isnan(ema_50_aligned[i]) or np.isnan(ema_200_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Trend conditions
        uptrend = ema_50_aligned[i] > ema_200_aligned[i]
        downtrend = ema_50_aligned[i] < ema_200_aligned[i]
        
        # VWAP bounce conditions
        # Long: price touches VWAP from above and bounces up
        vwap_touch_above = low[i] <= vwap[i] and close[i] > vwap[i]
        # Short: price touches VWAP from below and bounces down
        vwap_touch_below = high[i] >= vwap[i] and close[i] < vwap[i]
        
        # Bollinger Band filter: only trade near bands in trending markets
        near_upper_bb = high[i] >= upper_bb[i] * 0.995  # within 0.5% of upper band
        near_lower_bb = low[i] <= lower_bb[i] * 1.005   # within 0.5% of lower band
        
        if position == 0:
            # Long: uptrend + VWAP bounce from above + near upper BB
            if uptrend and vwap_touch_above and near_upper_bb:
                signals[i] = 0.25
                position = 1
            # Short: downtrend + VWAP bounce from below + near lower BB
            elif downtrend and vwap_touch_below and near_lower_bb:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Long exit: trend change or VWAP crosses below
            if not uptrend or close[i] < vwap[i]:
                signals[i] = -0.25  # reverse to short
                position = -1
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: trend change or VWAP crosses above
            if not downtrend or close[i] > vwap[i]:
                signals[i] = 0.25  # reverse to long
                position = 1
            else:
                signals[i] = -0.25
    
    return signals

name = "4h_VWAP_Bounce_BullBear_v1"
timeframe = "4h"
leverage = 1.0
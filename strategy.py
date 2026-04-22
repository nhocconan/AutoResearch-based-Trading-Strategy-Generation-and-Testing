# -*- coding: utf-8 -*-
#!/usr/bin/env python3
"""
Hypothesis: 6h Williams Alligator + Elder Ray power with 1d trend filter.
- Williams Alligator (Jaws/Teeth/Lips) identifies trend phases via SMAs.
- Elder Ray (Bull/Bear Power = Close - EMA13) measures trend strength.
- Trade only when Alligator is aligned (bullish/bearish) AND Elder Ray confirms strength.
- Use 1d EMA50 as higher-timeframe trend filter to avoid counter-trend trades.
- Target: 50-150 total trades over 4 years (~12-37/year) with size 0.25.
- Works in bull (riding trends) and bear (avoiding false breaks via 1d filter).
"""

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
    
    # Load 1d data for trend filter (ONCE before loop)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # 1d EMA50 for trend filter
    close_1d = df_1d['close'].values
    close_1d_series = pd.Series(close_1d)
    ema_50_1d = close_1d_series.ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # Williams Alligator on 6h: SMAs of median price
    median_price = (high + low) / 2
    jaw = pd.Series(median_price).rolling(window=13, min_periods=13).mean().values  # 13-period
    teeth = pd.Series(median_price).rolling(window=8, min_periods=8).mean().values    # 8-period
    lips = pd.Series(median_price).rolling(window=5, min_periods=5).mean().values     # 5-period
    
    # Elder Ray: Bull Power = Close - EMA13, Bear Power = EMA13 - Close
    ema_13 = pd.Series(close).ewm(span=13, adjust=False, min_periods=13).mean().values
    bull_power = close - ema_13
    bear_power = ema_13 - close
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(13, n):  # Wait for Alligator components
        # Skip if 1d trend filter not ready
        if np.isnan(ema_50_1d_aligned[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Alligator alignment: check if jaws < teeth < lips (bullish) or jaws > teeth > lips (bearish)
        if (jaw[i] < teeth[i] < lips[i]):  # Bullish alignment
            bullish_aligned = True
        elif (jaw[i] > teeth[i] > lips[i]):  # Bearish alignment
            bullish_aligned = False
        else:
            bullish_aligned = None  # No clear alignment (intertwined)
        
        if position == 0 and bullish_aligned is not None:
            # Enter long: bullish Alligator + positive Bull Power + above 1d EMA50
            if bullish_aligned and bull_power[i] > 0 and close[i] > ema_50_1d_aligned[i]:
                signals[i] = 0.25
                position = 1
            # Enter short: bearish Alligator + positive Bear Power + below 1d EMA50
            elif not bullish_aligned and bear_power[i] > 0 and close[i] < ema_50_1d_aligned[i]:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: Alligator loses bullish alignment OR Bear Power becomes positive
            if not (jaw[i] < teeth[i] < lips[i]) or bear_power[i] > 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: Alligator loses bearish alignment OR Bull Power becomes positive
            if not (jaw[i] > teeth[i] > lips[i]) or bull_power[i] > 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "6H_Alligator_ElderRay_1dEMA50_Trend"
timeframe = "6h"
leverage = 1.0
#!/usr/bin/env python3
"""
6h_ElderRay_ZeroLag_EMA_Trend_Filter_v3
Hypothesis: 6h Elder Ray (Bull/Bear Power) with ZeroLag EMA trend filter and volume confirmation.
Elder Ray measures bull/bear power relative to EMA13. ZeroLag EMA reduces lag for timely trend detection.
Enter long when Bull Power > 0 AND Bear Power < 0 AND close > ZeroLag EMA50 AND volume > 1.5 * 20-period average.
Enter short when Bull Power < 0 AND Bear Power > 0 AND close < ZeroLag EMA50 AND volume > 1.5 * 20-period average.
Exit when Elder Ray signals reverse or volume drops below average.
Uses 12h EMA200 for higher timeframe trend alignment to avoid counter-trend trades in bear markets.
Designed for 6h timeframe to target 12-37 trades/year (50-150 total over 4 years).
Works in bull/bear markets by trading with the 12h trend and using Elder Ray to measure conviction.
Volume spike filters weak breakouts. ZeroLag EMA provides timely trend signals without excessive lag.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Load 12h data ONCE before loop for HTF trend filter
    df_12h = get_htf_data(prices, '12h')
    
    # Calculate 12h EMA200 for trend filter
    close_12h = df_12h['close'].values
    ema200_12h = pd.Series(close_12h).ewm(span=200, adjust=False, min_periods=200).mean().values
    ema200_12h_aligned = align_htf_to_ltf(prices, df_12h, ema200_12h)
    
    # Calculate Elder Ray components: need EMA13 for Bull/Bear Power
    ema13 = pd.Series(close).ewm(span=13, adjust=False, min_periods=13).mean().values
    bull_power = high - ema13  # Bull Power = High - EMA13
    bear_power = low - ema13   # Bear Power = Low - EMA13
    
    # Calculate ZeroLag EMA50 for entry timing
    # ZeroLag EMA = EMA + (EMA - EMA of EMA) to reduce lag
    ema50 = pd.Series(close).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema50_of_ema50 = pd.Series(ema50).ewm(span=50, adjust=False, min_periods=50).mean().values
    zl_ema50 = 2 * ema50 - ema50_of_ema50
    
    # Volume spike: volume > 1.5 * 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (1.5 * vol_ma)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after warmup (need 200 for 12h EMA, 50 for ZeroLag EMA, 13 for Elder Ray, 20 for volume MA)
    start_idx = max(200, 50, 13, 20)
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(ema200_12h_aligned[i]) or 
            np.isnan(zl_ema50[i]) or
            np.isnan(bull_power[i]) or np.isnan(bear_power[i]) or
            np.isnan(vol_ma[i])):
            # Hold current position
            if position == 0:
                signals[i] = 0.0
            elif position == 1:
                signals[i] = 0.25
            else:
                signals[i] = -0.25
            continue
        
        # Elder Ray conditions
        bull_positive = bull_power[i] > 0
        bear_negative = bear_power[i] < 0
        bull_negative = bull_power[i] < 0
        bear_positive = bear_power[i] > 0
        
        # Trend filter: close vs ZeroLag EMA50
        above_zl_ema = close[i] > zl_ema50[i]
        below_zl_ema = close[i] < zl_ema50[i]
        
        # Higher timeframe trend filter: close vs 12h EMA200
        above_12h_ema = close[i] > ema200_12h_aligned[i]
        below_12h_ema = close[i] < ema200_12h_aligned[i]
        
        if position == 0:
            # Long: Bull Power > 0 AND Bear Power < 0 AND above ZeroLag EMA50 AND above 12h EMA200 AND volume spike
            if bull_positive and bear_negative and above_zl_ema and above_12h_ema and volume_spike[i]:
                signals[i] = 0.25
                position = 1
            # Short: Bull Power < 0 AND Bear Power > 0 AND below ZeroLag EMA50 AND below 12h EMA200 AND volume spike
            elif bull_negative and bear_positive and below_zl_ema and below_12h_ema and volume_spike[i]:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # Hold long
            signals[i] = 0.25
            # Exit: Elder Ray reverses OR volume drops below average OR close crosses below ZeroLag EMA50
            if (bull_power[i] <= 0 or bear_power[i] >= 0 or not volume_spike[i] or close[i] <= zl_ema50[i]):
                signals[i] = 0.0
                position = 0
        elif position == -1:
            # Hold short
            signals[i] = -0.25
            # Exit: Elder Ray reverses OR volume drops below average OR close crosses above ZeroLag EMA50
            if (bull_power[i] >= 0 or bear_power[i] <= 0 or not volume_spike[i] or close[i] >= zl_ema50[i]):
                signals[i] = 0.0
                position = 0
    
    return signals

name = "6h_ElderRay_ZeroLag_EMA_Trend_Filter_v3"
timeframe = "6h"
leverage = 1.0
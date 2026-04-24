#!/usr/bin/env python3
"""
Hypothesis: 6h Williams %R with 12h EMA trend filter and volume spike confirmation.
- Primary timeframe: 6h for execution, HTF: 12h for EMA trend direction.
- Williams %R(14) on 6h: oversold < -80, overbought > -20.
- 12h EMA50: price above EMA50 = uptrend, below = downtrend.
- Entry logic: In uptrend (price > EMA50), go long when Williams %R crosses above -80 from below.
               In downtrend (price < EMA50), go short when Williams %R crosses below -20 from above.
- Exit: Opposite Williams %R crossover or EMA trend flip.
- Volume confirmation: current 6h volume > 2.0 * 20-period volume MA (filters low-activity bars).
- Discrete signal size: 0.25 to limit drawdown and reduce fee churn.
- Target: 50-150 total trades over 4 years (12-37/year) for 6h timeframe.
- Works in both bull and bear markets: trend filter ensures we only take trades in direction of 12h trend,
  while Williams %R provides mean-reversion entries within the trend.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Extract price and volume data
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 12h data for EMA50 trend filter
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 50:
        return np.zeros(n)
    
    # Calculate 12h EMA50
    ema_50 = pd.Series(df_12h['close']).ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Align 12h EMA50 to 6h
    ema_50_aligned = align_htf_to_ltf(prices, df_12h, ema_50)
    
    # Calculate Williams %R on 6h (14-period)
    # Williams %R = (Highest High - Close) / (Highest High - Lowest Low) * -100
    highest_high = pd.Series(high).rolling(window=14, min_periods=14).max().values
    lowest_low = pd.Series(low).rolling(window=14, min_periods=14).min().values
    williams_r = (highest_high - close) / (highest_high - lowest_low + 1e-10) * -100
    
    # Volume confirmation: current volume > 2.0 * 20-period volume MA
    volume_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (2.0 * volume_ma)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start from index where all indicators are ready
    start_idx = max(50, 20, 14)  # Need enough bars for EMA50, volume MA, Williams %R
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(ema_50_aligned[i]) or np.isnan(williams_r[i]) or 
            np.isnan(volume_spike[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        price = close[i]
        ema_trend = ema_50_aligned[i]
        wr = williams_r[i]
        vol_ok = volume_spike[i]
        
        if position == 0:
            # Check for entry signals with volume confirmation
            if vol_ok:
                # Uptrend: price above 12h EMA50
                if price > ema_trend:
                    # Long when Williams %R crosses above -80 from below (oversold bounce)
                    if wr > -80 and williams_r[i-1] <= -80:
                        signals[i] = 0.25
                        position = 1
                # Downtrend: price below 12h EMA50
                elif price < ema_trend:
                    # Short when Williams %R crosses below -20 from above (overbought rejection)
                    if wr < -20 and williams_r[i-1] >= -20:
                        signals[i] = -0.25
                        position = -1
        elif position == 1:
            # Long exit: Williams %R crosses below -50 (momentum loss) or trend flip to downtrend
            if wr < -50 or price < ema_trend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: Williams %R crosses above -50 (momentum loss) or trend flip to uptrend
            if wr > -50 or price > ema_trend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "6h_WilliamsR_12hEMA50_Trend_VolumeSpike_v1"
timeframe = "6h"
leverage = 1.0
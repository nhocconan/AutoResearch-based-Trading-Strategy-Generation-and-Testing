#!/usr/bin/env python3
"""
Hypothesis: 6h Williams Alligator with 1w EMA50 trend filter and volume confirmation.
- Long when Alligator jaws < teeth < lips (bullish alignment) AND price > lips AND 1w close > 1w EMA50 AND volume > 1.5 * 20-period average
- Short when Alligator jaws > teeth > lips (bearish alignment) AND price < lips AND 1w close < 1w EMA50 AND volume > 1.5 * 20-period average
- Exit on opposite Alligator signal (long exit when jaws > teeth, short exit when jaws < teeth)
- Uses 6h primary with 1w HTF to target 50-150 total trades over 4 years (12-37/year)
- Alligator identifies trend alignment; EMA50 filters weekly regime; volume confirms momentum
- Designed to work in both bull (trend following with alignment) and bear (counter-trend against alignment) markets
- Signal size: 0.25 discrete levels to minimize fee churn
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
    
    # Calculate Williams Alligator (SMMA = smoothed moving average)
    def smma(data, period):
        result = np.full_like(data, np.nan, dtype=np.float64)
        if len(data) < period:
            return result
        # First value is SMA
        result[period-1] = np.mean(data[:period])
        # Subsequent values: SMMA = (PREV_SMMA * (period-1) + CLOSE) / period
        for i in range(period, len(data)):
            result[i] = (result[i-1] * (period-1) + data[i]) / period
        return result
    
    # Alligator lines: jaws (13,8), teeth (8,5), lips (5,3)
    jaws = smma(close, 13)
    teeth = smma(close, 8)
    lips = smma(close, 5)
    
    # Calculate 1w EMA50 for trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 2:
        return np.zeros(n)
    
    ema_50_1w = pd.Series(df_1w['close'].values).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_50_1w)
    
    # Trend filter: bullish if close > EMA50, bearish if close < EMA50
    bullish_regime = close > ema_50_1w_aligned
    bearish_regime = close < ema_50_1w_aligned
    
    # Volume confirmation: volume > 1.5 * 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_confirm = volume > (1.5 * vol_ma)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start from index where all indicators are ready
    start_idx = max(13, 50, 20) + 1  # Need Alligator, EMA50, and volume MA
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(jaws[i]) or np.isnan(teeth[i]) or np.isnan(lips[i]) or 
            np.isnan(ema_50_1w_aligned[i]) or np.isnan(volume_confirm[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: bullish alignment AND price > lips AND bullish regime AND volume confirmation
            if jaws[i] < teeth[i] and teeth[i] < lips[i] and close[i] > lips[i] and bullish_regime[i] and volume_confirm[i]:
                signals[i] = 0.25
                position = 1
            # Short: bearish alignment AND price < lips AND bearish regime AND volume confirmation
            elif jaws[i] > teeth[i] and teeth[i] > lips[i] and close[i] < lips[i] and bearish_regime[i] and volume_confirm[i]:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Long exit: bearish alignment (jaws > teeth) - trend weakening
            if jaws[i] > teeth[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: bullish alignment (jaws < teeth) - trend weakening
            if jaws[i] < teeth[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "6h_Williams_Alligator_1wEMA50_VolumeConfirm_v1"
timeframe = "6h"
leverage = 1.0
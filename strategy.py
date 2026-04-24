#!/usr/bin/env python3
"""
Hypothesis: 4h Williams %R reversal with 12h EMA50 trend filter and volume spike confirmation.
- Long when Williams %R crosses above -80 (oversold reversal) AND 12h close > 12h EMA50 (bullish trend) AND volume > 1.5 * 20-period average
- Short when Williams %R crosses below -20 (overbought reversal) AND 12h close < 12h EMA50 (bearish trend) AND volume > 1.5 * 20-period average
- Exit when Williams %R crosses below -50 for longs or above -50 for shorts (mid-point reversion)
- Uses 4h primary with 12h HTF to target 75-200 total trades over 4 years (19-50/year)
- Williams %R captures momentum reversals; EMA50 filters trend; volume spike confirms strength
- Designed to work in both bull (buy oversold dips in uptrend) and bear (sell overbought rallies in downtrend) markets
- Signal size: 0.25 discrete levels to minimize fee churn
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
    
    # Calculate Williams %R: %R = (Highest High - Close) / (Highest High - Lowest Low) * -100
    # Using 14-period lookback
    period = 14
    highest_high = pd.Series(high).rolling(window=period, min_periods=period).max().values
    lowest_low = pd.Series(low).rolling(window=period, min_periods=period).min().values
    williams_r = (highest_high - close) / (highest_high - lowest_low) * -100
    # Handle division by zero when highest_high == lowest_low
    williams_r = np.where((highest_high - lowest_low) == 0, -50, williams_r)
    
    # Calculate 12h EMA50 for trend filter
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 2:
        return np.zeros(n)
    
    ema_50_12h = pd.Series(df_12h['close'].values).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_12h_aligned = align_htf_to_ltf(prices, df_12h, ema_50_12h)
    
    # Trend filter: bullish if close > EMA50, bearish if close < EMA50
    bullish_regime = close > ema_50_12h_aligned
    bearish_regime = close < ema_50_12h_aligned
    
    # Volume confirmation: volume > 1.5 * 20-period average (volume spike)
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_confirm = volume > (1.5 * vol_ma)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start from index where all indicators are ready
    start_idx = max(50, 20, 14) + 1  # Need EMA50, volume MA, and Williams %R
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(williams_r[i]) or np.isnan(ema_50_12h_aligned[i]) or 
            np.isnan(volume_confirm[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: Williams %R crosses above -80 (from below) AND bullish regime AND volume confirmation
            if williams_r[i] > -80 and williams_r[i-1] <= -80 and bullish_regime[i] and volume_confirm[i]:
                signals[i] = 0.25
                position = 1
            # Short: Williams %R crosses below -20 (from above) AND bearish regime AND volume confirmation
            elif williams_r[i] < -20 and williams_r[i-1] >= -20 and bearish_regime[i] and volume_confirm[i]:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Long exit: Williams %R crosses below -50 (from above)
            if williams_r[i] < -50 and williams_r[i-1] >= -50:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: Williams %R crosses above -50 (from below)
            if williams_r[i] > -50 and williams_r[i-1] <= -50:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "4h_WilliamsR_12hEMA50_VolumeSpike_v1"
timeframe = "4h"
leverage = 1.0
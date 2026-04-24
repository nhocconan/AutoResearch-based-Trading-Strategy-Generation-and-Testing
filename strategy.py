#!/usr/bin/env python3
"""
Hypothesis: 12h Williams %R reversal with 1w EMA50 trend filter and volume spike confirmation.
- Long when Williams %R crosses above -80 (oversold) AND 1w close > 1w EMA50 (bullish regime) AND volume > 1.5 * 20-period average
- Short when Williams %R crosses below -20 (overbought) AND 1w close < 1w EMA50 (bearish regime) AND volume > 1.5 * 20-period average
- Exit when Williams %R crosses -50 (mean reversion) or opposite regime condition fails
- Uses 12h primary with 1w HTF to target 50-150 trades over 4 years (12-37/year)
- Williams %R captures momentum reversals; 1w EMA50 filters regime; volume spike confirms conviction
- Designed to work in both bull (buy dips in uptrend) and bear (sell rallies in downtrend) markets
- Signal size: 0.25 discrete levels to minimize fee churn
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
    
    # Calculate Williams %R: %R = (Highest High - Close) / (Highest High - Lowest Low) * -100
    # Using 14-period lookback
    highest_high = pd.Series(high).rolling(window=14, min_periods=14).max().values
    lowest_low = pd.Series(low).rolling(window=14, min_periods=14).min().values
    williams_r = (highest_high - close) / (highest_high - lowest_low) * -100
    # Handle division by zero (when high == low)
    williams_r = np.where((highest_high - lowest_low) == 0, -50, williams_r)
    
    # Calculate 1w EMA50 for trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 50:
        return np.zeros(n)
    
    weekly_close = df_1w['close'].values
    ema_50_1w = pd.Series(weekly_close).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_50_1w)
    
    # Trend filter: bullish if close > EMA50, bearish if close < EMA50
    bullish_regime = close > ema_50_1w_aligned
    bearish_regime = close < ema_50_1w_aligned
    
    # Volume confirmation: volume > 1.5 * 20-period average (moderate spike)
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_confirm = volume > (1.5 * vol_ma)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start from index where all indicators are ready
    start_idx = max(14, 50, 20)  # Need Williams %R (14), EMA50 (50), and volume MA (20)
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(williams_r[i]) or np.isnan(ema_50_1w_aligned[i]) or 
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
            # Long exit: Williams %R crosses above -50 (from below) OR regime turns bearish
            if williams_r[i] > -50 and williams_r[i-1] <= -50:
                signals[i] = 0.0
                position = 0
            elif not bullish_regime[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: Williams %R crosses below -50 (from above) OR regime turns bullish
            if williams_r[i] < -50 and williams_r[i-1] >= -50:
                signals[i] = 0.0
                position = 0
            elif not bearish_regime[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "12h_WilliamsR_1wEMA50_VolumeSpike_v1"
timeframe = "12h"
leverage = 1.0
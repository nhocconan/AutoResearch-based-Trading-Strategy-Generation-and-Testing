#!/usr/bin/env python3
"""
Hypothesis: 12h Camarilla H4/L4 breakout with 1w EMA50 trend filter and volume confirmation.
- Long when price breaks above H4 level AND 1w EMA50 is rising AND volume > 1.3 * 20-period average
- Short when price breaks below L4 level AND 1w EMA50 is falling AND volume > 1.3 * 20-period average
- Exit when price returns to the Camarilla H3/L3 levels (mean reversion within the range)
- Uses 12h primary with 1w HTF for trend filter to avoid counter-trend trades
- Camarilla levels provide intraday support/resistance; EMA50 filters trend direction; volume confirms breakout strength
- Works in bull markets (breakouts with trend) and bear markets (breakdowns with trend)
- Signal size: 0.25 discrete levels to minimize fee churn
- Target: 50-150 total trades over 4 years (12-37/year)
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
    
    # Calculate Camarilla levels for 12h (using previous bar's OHLC)
    # H4 = close + 1.5*(high-low), L4 = close - 1.5*(high-low)
    # H3 = close + 1.25*(high-low), L3 = close - 1.25*(high-low)
    # We use previous bar to avoid look-ahead
    prev_close = np.roll(close, 1)
    prev_high = np.roll(high, 1)
    prev_low = np.roll(low, 1)
    prev_close[0] = close[0]  # First bar uses current values
    prev_high[0] = high[0]
    prev_low[0] = low[0]
    
    range_ = prev_high - prev_low
    H4 = prev_close + 1.5 * range_
    L4 = prev_close - 1.5 * range_
    H3 = prev_close + 1.25 * range_
    L3 = prev_close - 1.25 * range_
    
    # Calculate 1w EMA50 for trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 50:  # Need enough data for EMA50
        return np.zeros(n)
    
    ema_50 = pd.Series(df_1w['close'].values).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_aligned = align_htf_to_ltf(prices, df_1w, ema_50)
    
    # EMA50 slope: rising if current > previous, falling if current < previous
    ema_50_rising = ema_50_aligned > np.roll(ema_50_aligned, 1)
    ema_50_falling = ema_50_aligned < np.roll(ema_50_aligned, 1)
    ema_50_rising[0] = False
    ema_50_falling[0] = False
    
    # Volume confirmation: volume > 1.3 * 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_confirm = volume > (1.3 * vol_ma)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start from index where all indicators are ready
    start_idx = max(20, 50) + 1  # Need volume MA and EMA50 data
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(H4[i]) or np.isnan(L4[i]) or np.isnan(H3[i]) or np.isnan(L3[i]) or
            np.isnan(ema_50_aligned[i]) or np.isnan(volume_confirm[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: price breaks above H4 AND EMA50 rising AND volume confirmation
            if close[i] > H4[i] and ema_50_rising[i] and volume_confirm[i]:
                signals[i] = 0.25
                position = 1
            # Short: price breaks below L4 AND EMA50 falling AND volume confirmation
            elif close[i] < L4[i] and ema_50_falling[i] and volume_confirm[i]:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Long exit: price returns to H3 (mean reversion)
            if close[i] <= H3[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: price returns to L3 (mean reversion)
            if close[i] >= L3[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "12h_Camarilla_H4L4_1wEMA50_VolumeConfirm_v1"
timeframe = "12h"
leverage = 1.0
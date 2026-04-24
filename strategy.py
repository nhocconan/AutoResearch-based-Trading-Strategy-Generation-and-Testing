#!/usr/bin/env python3
"""
Hypothesis: 12h Camarilla H4L4 breakout with 1d EMA34 trend filter and volume spike confirmation.
- Long when price breaks above H4 level AND close > 1d EMA34 AND volume > 2.0 * 20-period average
- Short when price breaks below L4 level AND close < 1d EMA34 AND volume > 2.0 * 20-period average
- Exit when price retreats to Pivot level (mean reversion to equilibrium)
- Uses 12h primary with 1d HTF for EMA trend filter to avoid counter-trend entries
- Camarilla levels provide institutional pivot points; volume confirms breakout strength
- Designed to work in bull markets (breakouts with volume) and bear markets (breakdowns with volume)
- Signal size: 0.25 discrete levels to minimize fee churn
- Target: 50-150 total trades over 4 years (12-37/year)
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
    
    # Calculate 12h Camarilla levels (using previous bar's OHLC)
    # H4 = Close + 1.5 * (High - Low)
    # L4 = Close - 1.5 * (High - Low)
    # Pivot = (High + Low + Close) / 3
    prev_high = np.roll(high, 1)
    prev_low = np.roll(low, 1)
    prev_close = np.roll(close, 1)
    prev_high[0] = high[0]
    prev_low[0] = low[0]
    prev_close[0] = close[0]
    
    camarilla_h4 = prev_close + 1.5 * (prev_high - prev_low)
    camarilla_l4 = prev_close - 1.5 * (prev_high - prev_low)
    camarilla_pivot = (prev_high + prev_low + prev_close) / 3
    
    # Calculate 1d EMA34 for trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 34:
        return np.zeros(n)
    
    ema_34_1d = pd.Series(df_1d['close'].values).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # Volume confirmation: volume > 2.0 * 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_confirm = volume > (2.0 * vol_ma)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start from index where all indicators are ready
    start_idx = max(20, 34) + 1  # Need volume MA and EMA data
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(camarilla_h4[i]) or np.isnan(camarilla_l4[i]) or 
            np.isnan(ema_34_1d_aligned[i]) or np.isnan(volume_confirm[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: price breaks above H4 AND close > 1d EMA34 AND volume confirmation
            if close[i] > camarilla_h4[i] and close[i] > ema_34_1d_aligned[i] and volume_confirm[i]:
                signals[i] = 0.25
                position = 1
            # Short: price breaks below L4 AND close < 1d EMA34 AND volume confirmation
            elif close[i] < camarilla_l4[i] and close[i] < ema_34_1d_aligned[i] and volume_confirm[i]:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Long exit: price retreats to Pivot level (mean reversion)
            if close[i] <= camarilla_pivot[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: price retreats to Pivot level (mean reversion)
            if close[i] >= camarilla_pivot[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "12h_Camarilla_H4L4_1dEMA34_VolumeSpike_v1"
timeframe = "12h"
leverage = 1.0
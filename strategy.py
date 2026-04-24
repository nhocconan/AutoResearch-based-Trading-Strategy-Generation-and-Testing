#!/usr/bin/env python3
"""
Hypothesis: 4h Williams %R with 1d EMA34 trend filter and volume spike confirmation.
- Williams %R: momentum oscillator measuring overbought/oversold levels.
  Long when %R crosses above -80 from below (oversold bounce) in uptrend.
  Short when %R crosses below -20 from above (overbought rejection) in downtrend.
- Trend filter: Only trade in direction of 1d EMA34 (long if price > EMA34, short if price < EMA34).
- Volume confirmation: current volume > 2.0x 20-period volume MA to ensure strong participation.
- Discrete signal size: 0.25 to limit drawdown and reduce fee churn.
- Target: 75-200 total trades over 4 years (19-50/year) for 4h timeframe.
- Works in bull via buying oversold bounces in uptrend, in bear via selling overbought rejections in downtrend.
- Williams %R provides timely reversal signals; EMA34 filters counter-trend noise; volume spike confirms conviction.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def williams_r(high, low, close, period):
    """Williams %R oscillator"""
    if len(high) < period:
        return np.full_like(high, np.nan, dtype=float)
    highest_high = pd.Series(high).rolling(window=period, min_periods=period).max().values
    lowest_low = pd.Series(low).rolling(window=period, min_periods=period).min().values
    # Williams %R = (Highest High - Close) / (Highest High - Lowest Low) * -100
    wr = (highest_high - close) / (highest_high - lowest_low + 1e-10) * -100
    return wr

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Extract price and volume data
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Get 1d data for EMA34 trend filter and Williams %R
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Williams %R (14-period) on 1d
    wr_1d = williams_r(high_1d, low_1d, close_1d, 14)
    
    # Align Williams %R to 4h
    wr_1d_aligned = align_htf_to_ltf(prices, df_1d, wr_1d)
    
    # 1d EMA34 for trend filter
    ema_34_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # Volume confirmation: current volume > 2.0 * 20-period volume MA
    volume_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (2.0 * volume_ma)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start from index where all indicators are ready
    start_idx = max(34, 20, 14)  # EMA34 + volume MA + Williams %R
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(wr_1d_aligned[i]) or np.isnan(ema_34_1d_aligned[i]) or np.isnan(volume_spike[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Only trade in direction of 1d EMA34 trend
            if close[i] > ema_34_1d_aligned[i]:  # Uptrend (price above EMA34)
                # Long when Williams %R crosses above -80 from below (oversold bounce)
                if i > 0 and not np.isnan(wr_1d_aligned[i-1]):
                    if wr_1d_aligned[i-1] <= -80 and wr_1d_aligned[i] > -80 and volume_spike[i]:
                        signals[i] = 0.25
                        position = 1
            elif close[i] < ema_34_1d_aligned[i]:  # Downtrend (price below EMA34)
                # Short when Williams %R crosses below -20 from above (overbought rejection)
                if i > 0 and not np.isnan(wr_1d_aligned[i-1]):
                    if wr_1d_aligned[i-1] >= -20 and wr_1d_aligned[i] < -20 and volume_spike[i]:
                        signals[i] = -0.25
                        position = -1
        elif position == 1:
            # Long exit: Williams %R crosses below -50 (momentum loss) or opposite signal
            if not np.isnan(wr_1d_aligned[i]) and wr_1d_aligned[i] < -50:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: Williams %R crosses above -50 (momentum loss) or opposite signal
            if not np.isnan(wr_1d_aligned[i]) and wr_1d_aligned[i] > -50:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "4h_WilliamsR_1dEMA34_Trend_VolumeSpike_v1"
timeframe = "4h"
leverage = 1.0
#!/usr/bin/env python3
"""
Hypothesis: 1h Camarilla H3/L3 breakout with 4h EMA34 trend filter and volume confirmation.
- Primary timeframe: 1h for execution, HTF: 4h for EMA trend and 1d for Camarilla pivot levels.
- Camarilla pivot: calculated from prior 1d OHLC. Long when price breaks above H3 with volume,
  short when breaks below L3 with volume.
- Trend filter: Only trade in direction of 4h EMA34 (long if EMA34 rising, short if falling).
- Volume confirmation: current volume > 2.0x 20-period volume MA to ensure strong participation.
- Session filter: trade only between 08:00-20:00 UTC to avoid low-liquidity hours.
- Discrete signal size: 0.20 to limit drawdown and reduce fee churn.
- Target: 60-150 total trades over 4 years (15-37/year) for 1h timeframe.
- Works in bull via buying H3 breakouts in uptrend, in bear via selling L3 breakouts in downtrend.
- Uses Camarilla levels which adapt to volatility and work well in ranging/breakout markets.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Extract price and volume data
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Get 1d data for Camarilla pivot levels (prior day OHLC)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    # Calculate Camarilla levels from prior 1d candle
    # H3, L3, H4, L4 levels
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Camarilla levels: based on prior day's range
    # H3 = close + 1.1*(high-low)/2
    # L3 = close - 1.1*(high-low)/2
    # H4 = close + 1.1*(high-low)
    # L4 = close - 1.1*(high-low)
    camarilla_H3 = close_1d + 1.1 * (high_1d - low_1d) / 2
    camarilla_L3 = close_1d - 1.1 * (high_1d - low_1d) / 2
    camarilla_H4 = close_1d + 1.1 * (high_1d - low_1d)
    camarilla_L4 = close_1d - 1.1 * (high_1d - low_1d)
    
    # Align Camarilla levels to 1h (use prior day's levels for current day trading)
    camarilla_H3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_H3)
    camarilla_L3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_L3)
    camarilla_H4_aligned = align_htf_to_ltf(prices, df_1d, camarilla_H4)
    camarilla_L4_aligned = align_htf_to_ltf(prices, df_1d, camarilla_L4)
    
    # Get 4h data for EMA34 trend filter
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 34:
        return np.zeros(n)
    
    close_4h = df_4h['close'].values
    # 4h EMA34 for trend filter
    ema_34_4h = pd.Series(close_4h).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_4h_aligned = align_htf_to_ltf(prices, df_4h, ema_34_4h)
    
    # Volume confirmation: current volume > 2.0 * 20-period volume MA
    volume_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (2.0 * volume_ma)
    
    # Session filter: 08:00-20:00 UTC
    hours = prices.index.hour  # prices.index is DatetimeIndex
    in_session = (hours >= 8) & (hours <= 20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start from index where all indicators are ready
    start_idx = max(34, 20)  # EMA34 + volume MA
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(camarilla_H3_aligned[i]) or np.isnan(camarilla_L3_aligned[i]) or
            np.isnan(ema_34_4h_aligned[i]) or np.isnan(volume_spike[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Session filter
        if not in_session[i]:
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Only trade in direction of 4h EMA34 trend
            if i > 0 and not np.isnan(ema_34_4h_aligned[i-1]):
                ema34_slope = ema_34_4h_aligned[i] - ema_34_4h_aligned[i-1]
                if ema34_slope > 0:  # Uptrend
                    # Long when price breaks above H3 with volume confirmation
                    if close[i] > camarilla_H3_aligned[i] and volume_spike[i]:
                        signals[i] = 0.20
                        position = 1
                elif ema34_slope < 0:  # Downtrend
                    # Short when price breaks below L3 with volume confirmation
                    if close[i] < camarilla_L3_aligned[i] and volume_spike[i]:
                        signals[i] = -0.20
                        position = -1
        elif position == 1:
            # Long exit: price breaks below L3 or opposite signal
            if close[i] < camarilla_L3_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.20
        elif position == -1:
            # Short exit: price breaks above H3 or opposite signal
            if close[i] > camarilla_H3_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.20
    
    return signals

name = "1h_Camarilla_H3L3_Breakout_4hEMA34_Trend_VolumeSpike_v1"
timeframe = "1h"
leverage = 1.0
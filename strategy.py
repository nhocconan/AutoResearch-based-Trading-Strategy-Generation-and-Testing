#!/usr/bin/env python3
"""
Hypothesis: 1h Camarilla H4/L4 breakout with 4h EMA20 trend filter and volume confirmation.
- Primary timeframe: 1h for execution, HTF: 4h for EMA trend and Camarilla pivots.
- Camarilla pivot levels (H4, L4) from prior 4h: Long when price > H4, Short when price < L4.
- Trend filter: Only trade in direction of 4h EMA20 (long if EMA20 rising, short if falling).
- Volume confirmation: current volume > 1.5x 20-period volume MA to ensure strong participation.
- Session filter: Only trade between 08:00-20:00 UTC to avoid low-liquidity hours.
- Discrete signal size: 0.20 to limit drawdown and reduce fee churn.
- Target: 60-150 total trades over 4 years (15-37/year) for 1h timeframe.
- Works in bull via buying breakouts in uptrend, in bear via selling breakdowns in downtrend.
- Uses actual Camarilla calculation: H4 = C + 1.1*(H-L), L4 = C - 1.1*(H-L).
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
    
    # Get 4h data for Camarilla pivots and EMA20
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 2:
        return np.zeros(n)
    
    high_4h = df_4h['high'].values
    low_4h = df_4h['low'].values
    close_4h = df_4h['close'].values
    
    # Calculate Camarilla levels from prior 4h bar
    # H4 = C + 1.1*(H-L), L4 = C - 1.1*(H-L)
    camarilla_H4 = close_4h + 1.1 * (high_4h - low_4h)
    camarilla_L4 = close_4h - 1.1 * (high_4h - low_4h)
    
    # Align to 1h: use prior 4h's levels (already completed bar)
    camarilla_H4_aligned = align_htf_to_ltf(prices, df_4h, camarilla_H4)
    camarilla_L4_aligned = align_htf_to_ltf(prices, df_4h, camarilla_L4)
    
    # 4h EMA20 for trend filter
    ema_20_4h = pd.Series(close_4h).ewm(span=20, adjust=False, min_periods=20).mean().values
    ema_20_4h_aligned = align_htf_to_ltf(prices, df_4h, ema_20_4h)
    
    # Volume confirmation: current volume > 1.5 * 20-period volume MA
    volume_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (1.5 * volume_ma)
    
    # Session filter: 08:00-20:00 UTC
    hours = prices.index.hour  # prices.index is DatetimeIndex
    in_session = (hours >= 8) & (hours <= 20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start from index where all indicators are ready
    start_idx = max(20, 20)  # EMA20 + volume MA
    
    for i in range(start_idx, n):
        # Skip if data not ready or outside session
        if (np.isnan(camarilla_H4_aligned[i]) or np.isnan(camarilla_L4_aligned[i]) or
            np.isnan(ema_20_4h_aligned[i]) or np.isnan(volume_spike[i]) or
            not in_session[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Only trade in direction of 4h EMA20 trend
            if i > 0 and not np.isnan(ema_20_4h_aligned[i-1]):
                ema20_slope = ema_20_4h_aligned[i] - ema_20_4h_aligned[i-1]
                if ema20_slope > 0:  # Uptrend
                    if close[i] > camarilla_H4_aligned[i] and volume_spike[i]:
                        # Buy on H4 breakout in uptrend
                        signals[i] = 0.20
                        position = 1
                elif ema20_slope < 0:  # Downtrend
                    if close[i] < camarilla_L4_aligned[i] and volume_spike[i]:
                        # Sell on L4 breakdown in downtrend
                        signals[i] = -0.20
                        position = -1
        elif position == 1:
            # Long exit: price returns to Camarilla H3/L3 level or opposite break
            # Calculate H3 and L3 for exit (from same prior 4h bar)
            camarilla_H3 = close_4h + 1.1 * (high_4h - low_4h) / 4  # H3 = C + 1.1*(H-L)/4
            camarilla_L3 = close_4h - 1.1 * (high_4h - low_4h) / 4  # L3 = C - 1.1*(H-L)/4
            camarilla_H3_aligned = align_htf_to_ltf(prices, df_4h, camarilla_H3)
            camarilla_L3_aligned = align_htf_to_ltf(prices, df_4h, camarilla_L3)
            
            if not np.isnan(camarilla_H3_aligned[i]) and not np.isnan(camarilla_L3_aligned[i]):
                if close[i] < camarilla_H3_aligned[i] or close[i] < camarilla_L4_aligned[i]:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = 0.20
        elif position == -1:
            # Short exit: price returns to Camarilla H3/L3 level or opposite break
            camarilla_H3 = close_4h + 1.1 * (high_4h - low_4h) / 4
            camarilla_L3 = close_4h - 1.1 * (high_4h - low_4h) / 4
            camarilla_H3_aligned = align_htf_to_ltf(prices, df_4h, camarilla_H3)
            camarilla_L3_aligned = align_htf_to_ltf(prices, df_4h, camarilla_L3)
            
            if not np.isnan(camarilla_H3_aligned[i]) and not np.isnan(camarilla_L3_aligned[i]):
                if close[i] > camarilla_L3_aligned[i] or close[i] > camarilla_H4_aligned[i]:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = -0.20
    
    return signals

name = "1h_Camarilla_H4L4_4hEMA20_Trend_VolumeSpike_v1"
timeframe = "1h"
leverage = 1.0
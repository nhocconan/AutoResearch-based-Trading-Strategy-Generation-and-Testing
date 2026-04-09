#!/usr/bin/env python3
# mtf_1h_ema_pullback_volume_4h1d_v1
# Hypothesis: 1h strategy trading pullbacks to EMA21 in the direction of 4h/1d trend.
# Uses 4h EMA50 for trend direction and 1d EMA200 for regime filter (avoid counter-trend in strong trends).
# Entry: price pulls back to 1h EMA21 with volume confirmation in direction of 4h/1d trend.
# Exit: opposite EMA21 cross or volume fade. Designed for low trade frequency (15-35/year) to minimize fee drag.
# Works in bull markets (trend continuation) and bear markets (trend continuation short).

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_1h_ema_pullback_volume_4h1d_v1"
timeframe = "1h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 200:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # 4h HTF for trend direction: EMA50
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 50:
        return np.zeros(n)
    
    close_4h = df_4h['close'].values
    ema_50_4h = pd.Series(close_4h).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_4h_aligned = align_htf_to_ltf(prices, df_4h, ema_50_4h)
    
    # 1d HTF for regime filter: EMA200
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 200:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    ema_200_1d = pd.Series(close_1d).ewm(span=200, adjust=False, min_periods=200).mean().values
    ema_200_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_200_1d)
    
    # 1h indicators: EMA21 for dynamic support/resistance, volume MA20 for confirmation
    close_s = pd.Series(close)
    ema_21 = close_s.ewm(span=21, adjust=False, min_periods=21).mean().values
    
    volume_s = pd.Series(volume)
    volume_ma20 = volume_s.rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(200, n):
        # Skip if any required data is NaN
        if (np.isnan(ema_50_4h_aligned[i]) or np.isnan(ema_200_1d_aligned[i]) or
            np.isnan(ema_21[i]) or np.isnan(volume_ma20[i])):
            signals[i] = 0.0
            continue
        
        # Volume confirmation: current volume > 1.3x 20-period average
        volume_confirmed = volume[i] > 1.3 * volume_ma20[i]
        
        if position == 1:  # Long position
            # Exit: price crosses below EMA21 OR volume dries up
            if close[i] < ema_21[i] or not volume_confirmed:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.20
                
        elif position == -1:  # Short position
            # Exit: price crosses above EMA21 OR volume dries up
            if close[i] > ema_21[i] or not volume_confirmed:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.20
        else:  # Flat
            if volume_confirmed:
                # Determine trend direction from 4h EMA50
                trend_up = close[i] > ema_50_4h_aligned[i]
                trend_down = close[i] < ema_50_4h_aligned[i]
                
                # Regime filter: avoid counter-trend trades when price is far from 1d EMA200
                price_vs_200ema = close[i] / ema_200_1d_aligned[i]
                extreme_bull = price_vs_200ema > 1.25   # Avoid longs in extreme bull
                extreme_bear = price_vs_200ema < 0.75   # Avoid shorts in extreme bear
                
                # Long setup: pullback to EMA21 in uptrend, not extreme bull
                if trend_up and close[i] <= ema_21[i] * 1.01 and not extreme_bull:
                    # Additional filter: price above 4h EMA50 (already in trend_up) and volume confirmation
                    position = 1
                    signals[i] = 0.20
                # Short setup: pullback to EMA21 in downtrend, not extreme bear
                elif trend_down and close[i] >= ema_21[i] * 0.99 and not extreme_bear:
                    # Additional filter: price below 4h EMA50 (already in trend_down) and volume confirmation
                    position = -1
                    signals[i] = -0.20
    
    return signals
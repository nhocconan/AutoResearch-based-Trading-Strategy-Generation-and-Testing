#!/usr/bin/env python3
"""
1h_Camarilla_R1_S1_Breakout_4hTrend_VolumeSpike
Hypothesis: 1h Camarilla R1/S1 breakout with 4h EMA50 trend filter and volume confirmation.
Long when price breaks above R1 in 4h bullish trend with volume spike.
Short when price breaks below S1 in 4h bearish trend with volume spike.
Camarilla levels provide intraday support/resistance, 4h EMA50 filters trend direction,
volume confirms institutional interest. Works in bull/bear by following 4h trend.
Discrete position sizing (0.20) minimizes fee churn. Targets 15-37 trades/year on 1h.
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
    
    # Session filter: 08-20 UTC (pre-compute)
    hours = prices.index.hour
    in_session = (hours >= 8) & (hours <= 20)
    
    # Get 4h data for trend filter
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 50:
        return np.zeros(n)
    
    # 4h EMA50 for trend
    close_4h = df_4h['close'].values
    ema_4h = pd.Series(close_4h).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_4h_aligned = align_htf_to_ltf(prices, df_4h, ema_4h)
    
    # Calculate 4h trend: price above/below EMA50
    trend_bullish = close_4h > ema_4h
    trend_bearish = close_4h < ema_4h
    trend_bullish_aligned = align_htf_to_ltf(prices, df_4h, trend_bullish.astype(float))
    trend_bearish_aligned = align_htf_to_ltf(prices, df_4h, trend_bearish.astype(float))
    
    # Calculate Camarilla levels from previous 1h bar
    # R1 = close + 1.1*(high-low)/12
    # S1 = close - 1.1*(high-low)/12
    # Using previous bar to avoid look-ahead
    prev_high = np.roll(high, 1)
    prev_low = np.roll(low, 1)
    prev_close = np.roll(close, 1)
    prev_high[0] = np.nan
    prev_low[0] = np.nan
    prev_close[0] = np.nan
    
    camarilla_range = prev_high - prev_low
    r1 = prev_close + 1.1 * camarilla_range / 12.0
    s1 = prev_close - 1.1 * camarilla_range / 12.0
    
    # Volume confirmation: volume > 1.5x 20-period MA
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (vol_ma * 1.5)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after warmup (need 50 for EMA, 20 for volume MA, 1 for Camarilla)
    start_idx = max(50, 20, 1)
    
    for i in range(start_idx, n):
        # Skip if outside session
        if not in_session[i]:
            if position == 0:
                signals[i] = 0.0
            elif position == 1:
                signals[i] = 0.20
            else:
                signals[i] = -0.20
            continue
        
        # Skip if any data not ready
        if (np.isnan(ema_4h_aligned[i]) or np.isnan(r1[i]) or np.isnan(s1[i]) or 
            np.isnan(volume_spike[i])):
            # Hold current position
            if position == 0:
                signals[i] = 0.0
            elif position == 1:
                signals[i] = 0.20
            else:
                signals[i] = -0.20
            continue
        
        if position == 0:
            # Long: price breaks above R1 with 4h bullish trend and volume spike
            if (close[i] > r1[i] and 
                trend_bullish_aligned[i] and volume_spike[i]):
                signals[i] = 0.20
                position = 1
            # Short: price breaks below S1 with 4h bearish trend and volume spike
            elif (close[i] < s1[i] and 
                  trend_bearish_aligned[i] and volume_spike[i]):
                signals[i] = -0.20
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # Hold long
            signals[i] = 0.20
            # Exit: price closes below S1 OR 4h trend turns bearish
            if (close[i] < s1[i] or not trend_bullish_aligned[i]):
                signals[i] = 0.0
                position = 0
        elif position == -1:
            # Hold short
            signals[i] = -0.20
            # Exit: price closes above R1 OR 4h trend turns bullish
            if (close[i] > r1[i] or not trend_bearish_aligned[i]):
                signals[i] = 0.0
                position = 0
    
    return signals

name = "1h_Camarilla_R1_S1_Breakout_4hTrend_VolumeSpike"
timeframe = "1h"
leverage = 1.0
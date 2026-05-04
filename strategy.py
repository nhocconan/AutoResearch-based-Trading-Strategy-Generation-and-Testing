#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Elder Ray (Bull/Bear Power) with 1d EMA50 trend filter and 1w volume spike confirmation
# Elder Ray measures bull/bear power relative to EMA13: Bull Power = High - EMA13, Bear Power = Low - EMA13
# Long when Bull Power > 0 and Bear Power rising (bullish momentum) + price > 1d EMA50 (uptrend) + 1w volume > 1.5x 20-period EMA
# Short when Bear Power < 0 and Bull Power falling (bearish momentum) + price < 1d EMA50 (downtrend) + 1w volume > 1.5x 20-period EMA
# Uses 1d EMA50 for primary trend (avoids whipsaw in ranging markets) and 1w volume spike to confirm institutional participation
# Discrete sizing (0.25) to minimize fee churn. Target: 12-30 trades/year on 6h timeframe.
# Works in bull markets via trend continuation and in bear markets via counter-trend reversals at extremes

name = "6h_ElderRay_1dEMA50_Trend_1wVolumeSpike"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Get 1d data for EMA50 trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 60:
        return np.zeros(n)
    
    # Get 1w data for volume confirmation
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 30:
        return np.zeros(n)
    
    # Calculate EMA13 for Elder Ray (using 6h data)
    close_series = pd.Series(close)
    ema13 = close_series.ewm(span=13, adjust=False, min_periods=13).mean().values
    
    # Elder Ray components
    bull_power = high - ema13   # Bull Power: High - EMA13
    bear_power = low - ema13    # Bear Power: Low - EMA13
    
    # Calculate 1d EMA50 for trend filter
    close_1d = df_1d['close'].values
    ema50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema50_1d)
    
    # Calculate 1w volume EMA20 for volume spike confirmation
    volume_1w = df_1w['volume'].values
    vol_ema20_1w = pd.Series(volume_1w).ewm(span=20, adjust=False, min_periods=20).mean().values
    vol_ema20_1w_aligned = align_htf_to_ltf(prices, df_1w, vol_ema20_1w)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(100, n):  # Start from 100 to have valid indicators
        # Skip if any value is NaN
        if (np.isnan(bull_power[i]) or np.isnan(bear_power[i]) or 
            np.isnan(ema50_1d_aligned[i]) or np.isnan(vol_ema20_1w_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Volume confirmation: current 1w volume > 1.5 x 20-period EMA (aligned to 6h)
        # Since we don't have direct 1w volume at 6h frequency, use aligned value as proxy
        volume_confirmed = volume[i] > (1.5 * vol_ema20_1w_aligned[i])
        
        if position == 0:
            # Long: Bull Power > 0 AND Bear Power rising (bullish momentum) + price > 1d EMA50 + volume confirmation
            # Bear Power rising: current > previous
            if (bull_power[i] > 0 and 
                bear_power[i] > bear_power[i-1] and 
                close[i] > ema50_1d_aligned[i] and 
                volume_confirmed):
                signals[i] = 0.25
                position = 1
            # Short: Bear Power < 0 AND Bull Power falling (bearish momentum) + price < 1d EMA50 + volume confirmation
            # Bull Power falling: current < previous
            elif (bear_power[i] < 0 and 
                  bull_power[i] < bull_power[i-1] and 
                  close[i] < ema50_1d_aligned[i] and 
                  volume_confirmed):
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: Bear Power > 0 (bullish momentum fading) OR price < 1d EMA50 (trend break)
            if bear_power[i] > 0 or close[i] < ema50_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: Bull Power < 0 (bearish momentum fading) OR price > 1d EMA50 (trend break)
            if bull_power[i] < 0 or close[i] > ema50_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals
#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h strategy using 1-day Parabolic SAR with volume and momentum confirmation
# Parabolic SAR identifies trend direction and reversal points with built-in acceleration
# Long when price > SAR and bullish momentum (close > open), short when price < SAR and bearish momentum
# Volume filter (>1.5x 20-period average) confirms breakout strength
# Works in bull/bear markets: SAR adapts quickly to trend changes, capturing both rallies and declines
# Target: 75-200 total trades over 4 years (19-50/year) with 0.25 position sizing

name = "4h_ParabolicSAR_VolumeMom_v1"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 30:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Calculate daily Parabolic SAR ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    
    if len(df_1d) < 3:
        return np.zeros(n)
    
    # Parabolic SAR parameters
    af_start = 0.02
    af_increment = 0.02
    af_max = 0.2
    
    # Arrays for SAR calculation
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Initialize SAR arrays
    sar = np.zeros_like(close_1d)
    trend = np.ones_like(close_1d)  # 1 for uptrend, -1 for downtrend
    af = np.full_like(close_1d, af_start)
    ep = np.zeros_like(close_1d)  # Extreme point
    
    # Set initial values
    sar[0] = low_1d[0]
    ep[0] = high_1d[0]
    
    # Calculate SAR for each day
    for i in range(1, len(close_1d)):
        if trend[i-1] == 1:  # uptrend
            sar[i] = sar[i-1] + af[i-1] * (ep[i-1] - sar[i-1])
            # SAR cannot exceed the low of the past two periods
            sar[i] = min(sar[i], low_1d[i-1], low_1d[i-2] if i >= 2 else low_1d[i-1])
            
            # Trend reversal check
            if low_1d[i] < sar[i]:
                trend[i] = -1
                sar[i] = ep[i-1]
                ep[i] = low_1d[i]
                af[i] = af_start
            else:
                trend[i] = 1
                if high_1d[i] > ep[i-1]:
                    ep[i] = high_1d[i]
                    af[i] = min(af[i-1] + af_increment, af_max)
                else:
                    ep[i] = ep[i-1]
                    af[i] = af[i-1]
        else:  # downtrend
            sar[i] = sar[i-1] + af[i-1] * (sar[i-1] - ep[i-1])
            # SAR cannot be below the high of the past two periods
            sar[i] = max(sar[i], high_1d[i-1], high_1d[i-2] if i >= 2 else high_1d[i-1])
            
            # Trend reversal check
            if high_1d[i] > sar[i]:
                trend[i] = 1
                sar[i] = ep[i-1]
                ep[i] = high_1d[i]
                af[i] = af_start
            else:
                trend[i] = -1
                if low_1d[i] < ep[i-1]:
                    ep[i] = low_1d[i]
                    af[i] = min(af[i-1] + af_increment, af_max)
                else:
                    ep[i] = ep[i-1]
                    af[i] = af[i-1]
    
    # Align daily SAR to 4h timeframe
    sar_aligned = align_htf_to_ltf(prices, df_1d, sar)
    trend_aligned = align_htf_to_ltf(prices, df_1d, trend)
    
    # Volume confirmation: >1.5x 20-period average
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_filter = volume > (1.5 * vol_ma_20)
    
    # Momentum confirmation: bullish/bearish candle
    bullish_momentum = close > prices['open'].values
    bearish_momentum = close < prices['open'].values
    
    # Pre-compute session filter (08-20 UTC)
    hours = pd.DatetimeIndex(prices["open_time"]).hour
    session_filter = (hours >= 8) & (hours <= 20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(30, n):
        # Skip if any critical value is NaN or outside session
        if (np.isnan(sar_aligned[i]) or np.isnan(trend_aligned[i]) or 
            np.isnan(volume_filter[i]) or np.isnan(bullish_momentum[i]) or
            not session_filter[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long entry: price above SAR, uptrend, volume confirmation, bullish momentum
            if close[i] > sar_aligned[i] and trend_aligned[i] == 1 and volume_filter[i] and bullish_momentum[i]:
                signals[i] = 0.25
                position = 1
            # Short entry: price below SAR, downtrend, volume confirmation, bearish momentum
            elif close[i] < sar_aligned[i] and trend_aligned[i] == -1 and volume_filter[i] and bearish_momentum[i]:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: price crosses below SAR (trend reversal)
            if close[i] < sar_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: price crosses above SAR (trend reversal)
            if close[i] > sar_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals
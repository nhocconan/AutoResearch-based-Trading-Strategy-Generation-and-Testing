#!/usr/bin/env python3
# 4h_Parabolic_SAR_Trend_Follower
# Hypothesis: Parabolic SAR captures trend direction effectively in both bull and bear markets.
# Combined with volume confirmation and weekly trend filter to avoid false signals.
# Parabolic SAR provides clear entry/exit signals with built-in trend following.
# Target: 20-40 trades/year per symbol to minimize fee drag while capturing major trends.

name = "4h_Parabolic_SAR_Trend_Follower"
timeframe = "4h"
leverage = 1.0

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
    
    # Get weekly data for trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 2:
        return np.zeros(n)
    
    # Calculate weekly EMA for trend filter (34-period)
    ema_34_1w = pd.Series(df_1w['close']).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_34_1w)
    
    # Parabolic SAR calculation
    # Initialize SAR values
    sar = np.zeros(n)
    trend = np.ones(n)  # 1 for uptrend, -1 for downtrend
    af = 0.02  # acceleration factor
    max_af = 0.2
    
    # Initialize first values
    sar[0] = low[0]
    trend[0] = 1
    ep = high[0]  # extreme point
    
    for i in range(1, n):
        if trend[i-1] == 1:  # uptrend
            sar[i] = sar[i-1] + af * (ep - sar[i-1])
            # SAR cannot be above the low of the past two periods
            sar[i] = min(sar[i], low[i-1], low[i-2] if i >= 2 else low[i-1])
            
            # Trend reversal check
            if low[i] < sar[i]:
                trend[i] = -1
                sar[i] = ep  # SAR becomes the previous EP
                ep = low[i]  # reset EP to current low
                af = 0.02    # reset acceleration factor
            else:
                trend[i] = 1
                if high[i] > ep:
                    ep = high[i]
                    af = min(af + 0.02, max_af)
                else:
                    trend[i] = trend[i-1]
                    ep = ep
                    af = af
        else:  # downtrend
            sar[i] = sar[i-1] - af * (sar[i-1] - ep)
            # SAR cannot be below the high of the past two periods
            sar[i] = max(sar[i], high[i-1], high[i-2] if i >= 2 else high[i-1])
            
            # Trend reversal check
            if high[i] > sar[i]:
                trend[i] = 1
                sar[i] = ep  # SAR becomes the previous EP
                ep = high[i]  # reset EP to current high
                af = 0.02    # reset acceleration factor
            else:
                trend[i] = -1
                if low[i] < ep:
                    ep = low[i]
                    af = min(af + 0.02, max_af)
                else:
                    trend[i] = trend[i-1]
                    ep = ep
                    af = af
    
    # Volume confirmation (20-period MA on 4h chart)
    volume_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Warmup: need weekly EMA (34), volume MA (20), and SAR calculation
    start_idx = max(34, 20)
    
    for i in range(start_idx, n):
        # Skip if any critical values are NaN
        if (np.isnan(ema_34_1w_aligned[i]) or np.isnan(sar[i]) or 
            np.isnan(volume_ma[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Weekly trend filter
        uptrend = close[i] > ema_34_1w_aligned[i]
        downtrend = close[i] < ema_34_1w_aligned[i]
        
        # Volume confirmation
        volume_confirm = volume[i] > volume_ma[i] * 1.5
        
        # SAR-based signals
        sar_bullish = close[i] > sar[i]
        sar_bearish = close[i] < sar[i]
        
        if position == 0:
            # Long entry: price above SAR + weekly uptrend + volume spike
            if sar_bullish and uptrend and volume_confirm:
                signals[i] = 0.25
                position = 1
            # Short entry: price below SAR + weekly downtrend + volume spike
            elif sar_bearish and downtrend and volume_confirm:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Long exit: price falls below SAR or weekly trend turns down
            if sar_bearish or not uptrend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: price rises above SAR or weekly trend turns up
            if sar_bullish or not downtrend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals
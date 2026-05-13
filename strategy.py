#!/usr/bin/env python3
"""
6h_Keltner_Channel_Pullback
Hypothesis: In trending markets, price pulls back to the 20-period EMA (middle of Keltner Channel) before continuing. The upper/lower bands (EMA ± 2*ATR) act as dynamic support/resistance. We enter long when price touches the lower band during an uptrend (EMA20 rising) and short when price touches the upper band during a downtrend (EMA20 falling). Weekly trend filter ensures we only trade with the higher timeframe trend, reducing false signals in ranging markets. Works in bull markets by catching pullbacks in uptrends and in bear markets by catching pullbacks in downtrends.
"""

name = "6h_Keltner_Channel_Pullback"
timeframe = "6h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtr_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Calculate EMA(20) - middle of Keltner Channel
    ema_period = 20
    ema = np.zeros_like(close)
    ema[:] = np.nan
    if len(close) >= ema_period:
        ema[ema_period-1] = np.mean(close[:ema_period])
        for i in range(ema_period, len(close)):
            ema[i] = (close[i] * 2 + ema[i-1] * (ema_period - 1)) / (ema_period + 1)
    
    # Calculate ATR(10) for Keltner Channel width
    atr_period = 10
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]
    
    atr = np.zeros_like(close)
    atr[:] = np.nan
    if len(tr) >= atr_period:
        atr[atr_period-1] = np.mean(tr[:atr_period])
        for i in range(atr_period, len(tr)):
            atr[i] = (atr[i-1] * (atr_period-1) + tr[i]) / atr_period
    
    # Keltner Channel bands: EMA ± 2*ATR
    kc_upper = ema + (2 * atr)
    kc_lower = ema - (2 * atr)
    
    # Get weekly data for trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 20:
        return np.zeros(n)
    
    close_1w = df_1w['close'].values
    
    # Weekly EMA(20) for trend filter
    ema_20_1w = np.zeros_like(close_1w)
    ema_20_1w[:] = np.nan
    if len(close_1w) >= 20:
        ema_20_1w[19] = np.mean(close_1w[:20])
        for i in range(20, len(close_1w)):
            ema_20_1w[i] = (close_1w[i] * 2 + ema_20_1w[i-1] * 19) / 21
    
    # Weekly trend: 1 = uptrend (price above EMA20), -1 = downtrend (price below EMA20)
    weekly_trend = np.zeros_like(close_1w)
    weekly_trend[:] = np.nan
    for i in range(len(close_1w)):
        if not np.isnan(ema_20_1w[i]):
            weekly_trend[i] = 1 if close_1w[i] > ema_20_1w[i] else -1
    
    # Align weekly trend to 6h timeframe
    weekly_trend_aligned = align_htf_to_ltf(prices, df_1w, weekly_trend)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(50, n):  # Start after warmup period
        # Skip if any required data is NaN
        if (np.isnan(ema[i]) or np.isnan(kc_upper[i]) or np.isnan(kc_lower[i]) or 
            np.isnan(weekly_trend_aligned[i])):
            signals[i] = 0.0
            continue
        
        if position == 0:
            # LONG: Price touches or crosses below lower KC band during weekly uptrend
            if (weekly_trend_aligned[i] == 1 and 
                low[i] <= kc_lower[i] and 
                close[i] > kc_lower[i]):  # Confirm with close above band
                signals[i] = 0.25
                position = 1
            # SHORT: Price touches or crosses above upper KC band during weekly downtrend
            elif (weekly_trend_aligned[i] == -1 and 
                  high[i] >= kc_upper[i] and 
                  close[i] < kc_upper[i]):  # Confirm with close below band
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: Price crosses back above EMA(20) or weekly trend turns down
            if (close[i] >= ema[i] or weekly_trend_aligned[i] == -1):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: Price crosses back below EMA(20) or weekly trend turns up
            if (close[i] <= ema[i] or weekly_trend_aligned[i] == 1):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals
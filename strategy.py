#!/usr/bin/env python3
"""
1d_Camarilla_R1_S1_Breakout_1wTrend_VolumeConfirmation
Hypothesis: On daily chart, price breaking above Camarilla R1 or below S1 with volume spike and aligned weekly trend (EMA34) provides high-probability trend-following entries. Works in bull/bear markets by following weekly trend. Uses volume confirmation to filter false breakouts and Camarilla levels for institutional-grade support/resistance. Target: 15-25 trades/year per symbol.
"""

name = "1d_Camarilla_R1_S1_Breakout_1wTrend_VolumeConfirmation"
timeframe = "1d"
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
    
    # Calculate Camarilla levels for today using yesterday's OHLC
    # Camarilla: R1 = C + (H-L)*1.1/12, S1 = C - (H-L)*1.1/12
    # We need previous day's OHLC, so shift by 1
    prev_high = np.roll(high, 1)
    prev_low = np.roll(low, 1)
    prev_close = np.roll(close, 1)
    prev_high[0] = high[0]  # fill first value
    prev_low[0] = low[0]
    prev_close[0] = close[0]
    
    # Camarilla R1 and S1 levels
    R1 = prev_close + (prev_high - prev_low) * 1.1 / 12
    S1 = prev_close - (prev_high - prev_low) * 1.1 / 12
    
    # Volume spike: today's volume > 1.5x 20-day average
    vol_ma = np.zeros(n)
    for i in range(20, n):
        vol_ma[i] = np.mean(volume[i-20:i])
    volume_spike = volume > (vol_ma * 1.5)
    
    # Breakout conditions
    breakout_long = (close > R1) & volume_spike
    breakout_short = (close < S1) & volume_spike
    
    # Get weekly data for trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 50:
        return np.zeros(n)
    
    close_1w = df_1w['close'].values
    
    # Weekly trend: 34 EMA
    ema_34_1w = pd.Series(close_1w).ewm(span=34, adjust=False, min_periods=34).mean().values
    uptrend_1w = close_1w > ema_34_1w
    downtrend_1w = close_1w < ema_34_1w
    
    # Align weekly trend to daily
    uptrend_1w_aligned = align_htf_to_ltf(prices, df_1w, uptrend_1w)
    downtrend_1w_aligned = align_htf_to_ltf(prices, df_1w, downtrend_1w)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(50, n):  # start after warmup period
        # Get aligned weekly trend
        uptrend = uptrend_1w_aligned[i]
        downtrend = downtrend_1w_aligned[i]
        
        if position == 0:
            # LONG: weekly uptrend + breakout above R1 with volume spike
            if uptrend and breakout_long[i]:
                signals[i] = 0.25
                position = 1
            # SHORT: weekly downtrend + breakout below S1 with volume spike
            elif downtrend and breakout_short[i]:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: weekly trend turns down OR price closes below EMA34 on daily
            if not uptrend or close[i] < ema_34_1w[i]:  # using weekly EMA for exit
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: weekly trend turns up OR price closes above EMA34 on daily
            if not downtrend or close[i] > ema_34_1w[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals
#!/usr/bin/env python3
# Hypothesis: 1h Camarilla R3/S3 breakout with 4h trend filter and volume spike.
# Enters long when price breaks above R3 level with 4h bullish trend (close > EMA20) and volume > 1.8x MA20.
# Enters short when price breaks below S3 level with 4h bearish trend (close < EMA20) and volume > 1.8x MA20.
# Exits when price reverts to the 1h EMA50 (adaptive mean reversion).
# Uses discrete position sizing (0.20) to minimize fee drag and manage drawdown.
# Designed for low trade frequency (~15-37/year) to work in both bull and bear markets by requiring volume confirmation and trend alignment.

name = "1h_Camarilla_R3S3_Breakout_4hTrend_Volume"
timeframe = "1h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 4h data for Camarilla pivot levels (based on previous 4h bar)
    df_4h = get_htf_data(prices, '4h')
    high_4h = df_4h['high'].values
    low_4h = df_4h['low'].values
    close_4h = df_4h['close'].values
    
    # Calculate Camarilla levels: R3, S3 (based on previous 4h bar)
    # R3 = close + 1.1*(high - low)/2
    # S3 = close - 1.1*(high - low)/2
    camarilla_r3 = close_4h + 1.1 * (high_4h - low_4h) / 2
    camarilla_s3 = close_4h - 1.1 * (high_4h - low_4h) / 2
    camarilla_r3_aligned = align_htf_to_ltf(prices, df_4h, camarilla_r3)
    camarilla_s3_aligned = align_htf_to_ltf(prices, df_4h, camarilla_s3)
    
    # Get 4h data for trend filter (EMA20)
    ema20_4h = pd.Series(close_4h).ewm(span=20, adjust=False, min_periods=20).mean().values
    ema20_4h_aligned = align_htf_to_ltf(prices, df_4h, ema20_4h)
    
    # Get 1h data for exit condition (EMA50)
    ema50_1h = pd.Series(close).ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Volume filter: current volume > 1.8x 20-period average (stricter to reduce trades)
    volume_series = pd.Series(volume)
    vol_ma20 = volume_series.rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (vol_ma20 * 1.8)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(100, n):  # Start after sufficient data for all indicators
        if np.isnan(camarilla_r3_aligned[i]) or np.isnan(camarilla_s3_aligned[i]) or \
           np.isnan(ema20_4h_aligned[i]) or np.isnan(vol_ma20[i]):
            signals[i] = 0.0
            continue
        
        if position == 0:
            # LONG: Price breaks above R3 with 4h bullish trend and volume spike
            if close[i] > camarilla_r3_aligned[i] and close[i] > ema20_4h_aligned[i] and volume_spike[i]:
                signals[i] = 0.20
                position = 1
            # SHORT: Price breaks below S3 with 4h bearish trend and volume spike
            elif close[i] < camarilla_s3_aligned[i] and close[i] < ema20_4h_aligned[i] and volume_spike[i]:
                signals[i] = -0.20
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: Price reverts to 1h EMA50 (mean reversion in range)
            if close[i] < ema50_1h[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.20
        elif position == -1:
            # EXIT SHORT: Price reverts to 1h EMA50 (mean reversion in range)
            if close[i] > ema50_1h[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.20
    
    return signals
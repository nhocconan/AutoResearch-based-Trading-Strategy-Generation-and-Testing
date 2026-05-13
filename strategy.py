#!/usr/bin/env python3
# Hypothesis: 12h Camarilla R3/S3 breakout with 1d EMA34 trend filter and volume confirmation.
# Long when price breaks above R3 with volume > 1.5x 20-period average AND close > 1d EMA34.
# Short when price breaks below S3 with volume > 1.5x 20-period average AND close < 1d EMA34.
# Uses discrete position sizing (0.25) to minimize fee churn. Designed for 12-37 trades/year.
# Works in bull markets via breakout momentum and in bear markets via breakdown strength.

name = "12h_Camarilla_R3S3_Breakout_1dTrend_Volume_v1"
timeframe = "12h"
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
    
    # Get 1d data for HTF trend filter
    df_1d = get_htf_data(prices, '1d')
    close_1d = df_1d['close'].values
    
    # Calculate EMA(34) on 1d close for trend filter
    ema34_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema34_1d)
    
    # Calculate Camarilla pivot levels for 12h
    # R3 = close + 1.1 * (high - low)
    # S3 = close - 1.1 * (high - low)
    camarilla_r3 = close + 1.1 * (high - low)
    camarilla_s3 = close - 1.1 * (high - low)
    
    # Volume confirmation: volume > 1.5x 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_confirm = volume > (1.5 * vol_ma)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(20, n):  # Start after sufficient data for volume MA
        if np.isnan(ema34_1d_aligned[i]) or np.isnan(camarilla_r3[i]) or np.isnan(camarilla_s3[i]) or \
           np.isnan(volume_confirm[i]):
            signals[i] = 0.0
            continue
        
        if position == 0:
            # LONG: Break above R3 with volume confirmation AND price > 1d EMA34
            if close[i] > camarilla_r3[i] and volume_confirm[i] and close[i] > ema34_1d_aligned[i]:
                signals[i] = 0.25
                position = 1
            # SHORT: Break below S3 with volume confirmation AND price < 1d EMA34
            elif close[i] < camarilla_s3[i] and volume_confirm[i] and close[i] < ema34_1d_aligned[i]:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: Price crosses below Camarilla pivot point (close) OR trend break
            pp = (high[i-1] + low[i-1] + close[i-1]) / 3.0  # Previous bar pivot point
            if close[i] < pp or close[i] < ema34_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: Price crosses above Camarilla pivot point (close) OR trend break
            pp = (high[i-1] + low[i-1] + close[i-1]) / 3.0  # Previous bar pivot point
            if close[i] > pp or close[i] > ema34_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals
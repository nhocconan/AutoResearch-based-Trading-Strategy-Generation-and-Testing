#!/usr/bin/env python3
# Hypothesis: 6h Camarilla R3/S3 breakout with 12h trend filter (EMA34) and volume confirmation (1.5x MA20).
# Enters long when price breaks above R3 level with 12h bullish trend (close > EMA34) and volume > 1.5x MA20.
# Enters short when price breaks below S3 level with 12h bearish trend (close < EMA34) and volume > 1.5x MA20.
# Exits when price crosses the 6h EMA20 (mean reversion).
# Uses discrete position sizing (0.25) to balance profit potential and drawdown control.
# Designed for low trade frequency (~12-37/year) by requiring confluence of breakout, trend, and volume.
# Works in both bull and bear markets: 12h trend filter ensures alignment with higher timeframe direction,
# while Camarilla R3/S3 levels provide significant breakout points with volume confirmation reducing false signals.

name = "6h_Camarilla_R3S3_Breakout_12hTrend_Volume"
timeframe = "6h"
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
    
    # Get 6h data for Camarilla pivot levels (based on previous 6h bar)
    df_6h = get_htf_data(prices, '6h')
    high_6h = df_6h['high'].values
    low_6h = df_6h['low'].values
    close_6h = df_6h['close'].values
    
    # Calculate Camarilla levels: R3, S3 (based on previous 6h bar)
    # R3 = close + 1.1*(high - low)/6
    # S3 = close - 1.1*(high - low)/6
    camarilla_r3 = close_6h + 1.1 * (high_6h - low_6h) / 6
    camarilla_s3 = close_6h - 1.1 * (high_6h - low_6h) / 6
    camarilla_r3_aligned = align_htf_to_ltf(prices, df_6h, camarilla_r3)
    camarilla_s3_aligned = align_htf_to_ltf(prices, df_6h, camarilla_s3)
    
    # Get 12h data for trend filter (EMA34)
    df_12h = get_htf_data(prices, '12h')
    close_12h = df_12h['close'].values
    ema34_12h = pd.Series(close_12h).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema34_12h_aligned = align_htf_to_ltf(prices, df_12h, ema34_12h)
    
    # Volume filter: current volume > 1.5x 20-period average
    volume_series = pd.Series(volume)
    vol_ma20 = volume_series.rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (vol_ma20 * 1.5)
    
    # 6h EMA20 for exit condition
    ema20_6h = pd.Series(close).ewm(span=20, adjust=False, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(100, n):  # Start after sufficient data for all indicators
        if np.isnan(camarilla_r3_aligned[i]) or np.isnan(camarilla_s3_aligned[i]) or \
           np.isnan(ema34_12h_aligned[i]) or np.isnan(vol_ma20[i]) or np.isnan(ema20_6h[i]):
            signals[i] = 0.0
            continue
        
        if position == 0:
            # LONG: Price breaks above R3 with 12h bullish trend and volume spike
            if close[i] > camarilla_r3_aligned[i] and close[i] > ema34_12h_aligned[i] and volume_spike[i]:
                signals[i] = 0.25
                position = 1
            # SHORT: Price breaks below S3 with 12h bearish trend and volume spike
            elif close[i] < camarilla_s3_aligned[i] and close[i] < ema34_12h_aligned[i] and volume_spike[i]:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: Price crosses below 6h EMA20 (mean reversion)
            if close[i] < ema20_6h[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: Price crosses above 6h EMA20 (mean reversion)
            if close[i] > ema20_6h[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals
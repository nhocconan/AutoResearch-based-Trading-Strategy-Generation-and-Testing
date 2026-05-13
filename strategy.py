#!/usr/bin/env python3
# Hypothesis: 6h Camarilla R3/S3 breakout with 1w trend filter (price > weekly SMA50 for longs, < for shorts) and volume confirmation (>1.5x 20-bar average).
# Uses discrete position sizing (0.25) to limit fee drag and drawdown.
# Weekly trend filter ensures alignment with major market direction, reducing whipsaws in ranging markets.
# Volume confirmation adds conviction to breakouts.
# Target: 50-150 total trades over 4 years (12-37/year) on 6h.

name = "6h_Camarilla_R3_S3_Breakout_1wSMA50_Trend_Volume_v1"
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
    
    # Calculate Camarilla pivot levels (based on previous bar)
    # R3 = close + 1.1*(high - low)/2
    # S3 = close - 1.1*(high - low)/2
    # R4 = close + 1.1*(high - low)
    # S4 = close - 1.1*(high - low)
    # Note: Using current bar's high/low for pivot calculation (standard for intraday)
    # For true Camarilla, should use previous bar, but we use current for simplicity and alignment
    # In practice, Camarilla uses previous day's OHLC, but for 6h we approximate with previous bar
    prev_close = np.roll(close, 1)
    prev_high = np.roll(high, 1)
    prev_low = np.roll(low, 1)
    prev_close[0] = close[0]  # First bar: use current close
    prev_high[0] = high[0]
    prev_low[0] = low[0]
    
    camarilla_range = prev_high - prev_low
    r3 = prev_close + 1.1 * camarilla_range / 2
    s3 = prev_close - 1.1 * camarilla_range / 2
    r4 = prev_close + 1.1 * camarilla_range
    s4 = prev_close - 1.1 * camarilla_range
    
    # Get 1w data for trend filter
    df_1w = get_htf_data(prices, '1w')
    close_1w = df_1w['close'].values
    
    # Calculate weekly SMA(50) for trend filter
    sma_50_1w = pd.Series(close_1w).rolling(window=50, min_periods=50).mean().values
    
    # Align weekly SMA50 to 6h timeframe (wait for weekly bar to close)
    sma_50_1w_aligned = align_htf_to_ltf(prices, df_1w, sma_50_1w)
    
    # Volume confirmation: volume > 1.5x 20-bar average
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_confirm = volume > (1.5 * vol_ma_20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(100, n):  # Start after sufficient data for indicators
        # Skip if any required data is NaN
        if (np.isnan(r3[i]) or np.isnan(s3[i]) or np.isnan(r4[i]) or np.isnan(s4[i]) or
            np.isnan(sma_50_1w_aligned[i]) or np.isnan(vol_ma_20[i])):
            signals[i] = 0.0
            continue
        
        if position == 0:
            # LONG: Price breaks above R3 AND weekly trend up (price > weekly SMA50) AND volume confirmation
            if close[i] > r3[i] and close[i] > sma_50_1w_aligned[i] and volume_confirm[i]:
                signals[i] = 0.25
                position = 1
            # SHORT: Price breaks below S3 AND weekly trend down (price < weekly SMA50) AND volume confirmation
            elif close[i] < s3[i] and close[i] < sma_50_1w_aligned[i] and volume_confirm[i]:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: Price breaks below S3 (mean reversion) OR weekly trend turns down
            if close[i] < s3[i] or close[i] < sma_50_1w_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: Price breaks above R3 (mean reversion) OR weekly trend turns up
            if close[i] > r3[i] or close[i] > sma_50_1w_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals
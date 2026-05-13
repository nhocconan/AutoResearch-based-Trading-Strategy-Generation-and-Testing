#!/usr/bin/env python3
# Hypothesis: 1h Camarilla R3/S3 breakout with 4h EMA50 trend filter and volume spike.
# Long when price breaks above R3 with volume > 1.5x 20-period average AND price > 4h EMA50.
# Short when price breaks below S3 with volume > 1.5x 20-period average AND price < 4h EMA50.
# Exit on opposite Camarilla level (R3/S3) or trend reversal (price crosses 4h EMA50).
# Uses discrete position sizing (0.20) to minimize fee churn. Target: 15-37 trades/year.
# Camarilla pivots work well in ranging markets (mean reversion at S1/R1, breakout at S3/R3).
# 4h EMA50 filters for trend alignment to avoid counter-trend breakouts.
# Volume spike confirms institutional participation in breakouts.
# 1h timeframe allows precise entry timing while 4h/1d HTF controls trade frequency.

name = "1h_Camarilla_R3S3_Breakout_4hTrend_Volume_v1"
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
    
    # Get 1h data for Camarilla pivot calculation (using typical price)
    typical_price = (high + low + close) / 3.0
    
    # Calculate Camarilla levels for 1h: based on previous bar's typical price
    # R3 = typical_price + 1.1 * (high - low)
    # S3 = typical_price - 1.1 * (high - low)
    typical_series = pd.Series(typical_price)
    high_series = pd.Series(high)
    low_series = pd.Series(low)
    
    # Shift by 1 to use previous bar's data (no look-ahead)
    prev_typical = typical_series.shift(1).values
    prev_high = high_series.shift(1).values
    prev_low = low_series.shift(1).values
    
    # Calculate Camarilla R3 and S3 levels
    camarilla_r3 = prev_typical + 1.1 * (prev_high - prev_low)
    camarilla_s3 = prev_typical - 1.1 * (prev_high - prev_low)
    
    # Get 4h data for EMA50 trend filter
    df_4h = get_htf_data(prices, '4h')
    close_4h = df_4h['close'].values
    
    # Calculate EMA(50) on 4h close for trend filter
    ema50_4h = pd.Series(close_4h).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema50_4h_aligned = align_htf_to_ltf(prices, df_4h, ema50_4h)
    
    # Volume filter: current volume > 1.5x 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_filter = volume > (1.5 * vol_ma)
    
    # Session filter: 08:00 to 20:00 UTC
    hours = prices.index.hour
    session_filter = (hours >= 8) & (hours <= 20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(100, n):  # Start after sufficient data for all indicators
        # Skip if any required data is NaN or outside session
        if (np.isnan(camarilla_r3[i]) or np.isnan(camarilla_s3[i]) or
            np.isnan(ema50_4h_aligned[i]) or np.isnan(vol_ma[i]) or
            not session_filter[i]):
            signals[i] = 0.0
            continue
        
        if position == 0:
            # LONG: price breaks above R3 with volume confirmation AND price > 4h EMA50
            if close[i] > camarilla_r3[i] and volume_filter[i] and close[i] > ema50_4h_aligned[i]:
                signals[i] = 0.20
                position = 1
            # SHORT: price breaks below S3 with volume confirmation AND price < 4h EMA50
            elif close[i] < camarilla_s3[i] and volume_filter[i] and close[i] < ema50_4h_aligned[i]:
                signals[i] = -0.20
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: price breaks below S3 OR trend reversal (price < 4h EMA50)
            if close[i] < camarilla_s3[i] or close[i] < ema50_4h_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.20
        elif position == -1:
            # EXIT SHORT: price breaks above R3 OR trend reversal (price > 4h EMA50)
            if close[i] > camarilla_r3[i] or close[i] > ema50_4h_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.20
    
    return signals
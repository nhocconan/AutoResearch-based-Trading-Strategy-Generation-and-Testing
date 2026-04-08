#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Strategy: 4h Camarilla Pivot + Volume + 1d Trend Filter
# Hypothesis: Daily Camarilla pivot levels act as strong support/resistance.
# Price rejection at R3/S3 with volume indicates reversal, breakout through R4/S4 with volume and trend alignment indicates continuation.
# 1d EMA50 filter ensures trades align with higher timeframe trend, working in both bull and bear markets.
# Targets 20-30 trades/year with disciplined entries to avoid overtrading.

name = "4h_camarilla_pivot_volume_trend_v1"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Price data
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # 1d Camarilla pivot levels (calculated from previous 1d bar)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    # Previous day's OHLC for pivot calculation
    prev_close = df_1d['close'].shift(1).values
    prev_high = df_1d['high'].shift(1).values
    prev_low = df_1d['low'].shift(1).values
    
    # Camarilla calculations
    pivot = (prev_high + prev_low + prev_close) / 3
    range_hl = prev_high - prev_low
    
    r3 = pivot + (range_hl * 1.1 / 2)
    s3 = pivot - (range_hl * 1.1 / 2)
    r4 = pivot + (range_hl * 1.1)
    s4 = pivot - (range_hl * 1.1)
    
    # Align to 4h timeframe
    r3_4h = align_htf_to_ltf(prices, df_1d, r3)
    s3_4h = align_htf_to_ltf(prices, df_1d, s3)
    r4_4h = align_htf_to_ltf(prices, df_1d, r4)
    s4_4h = align_htf_to_ltf(prices, df_1d, s4)
    
    # 1d EMA50 for trend filter
    ema50_1d = pd.Series(df_1d['close'].values).ewm(span=50, adjust=False).mean().values
    ema50_4h = align_htf_to_ltf(prices, df_1d, ema50_1d)
    
    # Volume confirmation: volume > 1.5x 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=10).mean().values
    vol_spike = volume > (1.5 * vol_ma)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(50, n):
        # Skip if required data not available
        if (np.isnan(r3_4h[i]) or np.isnan(s3_4h[i]) or 
            np.isnan(r4_4h[i]) or np.isnan(s4_4h[i]) or
            np.isnan(ema50_4h[i]) or np.isnan(vol_ma[i])):
            signals[i] = 0.0
            continue
        
        if position == 1:  # Long position
            # Exit: price crosses below S3 OR trend turns bearish
            if close[i] < s3_4h[i] or close[i] < ema50_4h[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
        elif position == -1:  # Short position
            # Exit: price crosses above R3 OR trend turns bullish
            if close[i] > r3_4h[i] or close[i] > ema50_4h[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat, look for entry
            # Fade at R3/S3 with volume spike
            if vol_spike[i]:
                # Sell at R3 rejection
                if close[i] < r3_4h[i] and (i == 50 or close[i-1] >= r3_4h[i-1]):
                    position = -1
                    signals[i] = -0.25
                # Buy at S3 rejection
                elif close[i] > s3_4h[i] and (i == 50 or close[i-1] <= s3_4h[i-1]):
                    position = 1
                    signals[i] = 0.25
            # Breakout through R4/S4 with volume and trend alignment
            if vol_spike[i]:
                # Buy breakout above R4 with bullish trend
                if close[i] > r4_4h[i] and (i == 50 or close[i-1] <= r4_4h[i-1]) and close[i] > ema50_4h[i]:
                    position = 1
                    signals[i] = 0.25
                # Sell breakout below S4 with bearish trend
                elif close[i] < s4_4h[i] and (i == 50 or close[i-1] >= s4_4h[i-1]) and close[i] < ema50_4h[i]:
                    position = -1
                    signals[i] = -0.25
    
    return signals
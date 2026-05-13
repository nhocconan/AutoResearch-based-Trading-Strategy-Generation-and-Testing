#!/usr/bin/env python3
# Hypothesis: 1h Camarilla R3/S3 breakout with 4h trend filter and volume confirmation.
# Long when price breaks above Camarilla R3 (1h) AND 4h close > 4h EMA50 AND 1h volume > 1.5x 20-bar average.
# Short when price breaks below Camarilla S3 (1h) AND 4h close < 4h EMA50 AND 1h volume > 1.5x 20-bar average.
# Exit when price returns to Camarilla Pivot point (1h) OR 4h EMA50 flips opposite direction.
# Uses Camarilla levels for precise intraday support/resistance, 4h EMA50 for trend filter, volume for confirmation.
# Session filter: 08-20 UTC to avoid low-liquidity hours.
# Position size: 0.20 (discrete levels to minimize fee churn).
# Target: 60-150 total trades over 4 years (15-37/year) by using tight entry conditions.
# Works in bull via breakout continuation, bear via faded rallies at key levels.

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
    
    # Pre-compute session filter (08-20 UTC)
    hours = prices.index.hour
    in_session = (hours >= 8) & (hours <= 20)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1h data for Camarilla calculation (need sufficient lookback)
    if n < 20:
        return np.zeros(n)
    
    # Calculate Camarilla levels for 1h using previous bar's OHLC
    # Camarilla: Pivot = (H+L+C)/3, Range = H-L
    # R3 = C + (H-L)*1.1/2, S3 = C - (H-L)*1.1/2
    prev_high = np.roll(high, 1)
    prev_low = np.roll(low, 1)
    prev_close = np.roll(close, 1)
    prev_high[0] = prev_high[1] if n > 1 else high[0]
    prev_low[0] = prev_low[1] if n > 1 else low[0]
    prev_close[0] = prev_close[1] if n > 1 else close[0]
    
    pivot = (prev_high + prev_low + prev_close) / 3.0
    range_hl = prev_high - prev_low
    r3 = pivot + (range_hl * 1.1 / 2.0)
    s3 = pivot - (range_hl * 1.1 / 2.0)
    
    # Get 4h data for EMA50 trend filter
    df_4h = get_htf_data(prices, '4h')
    close_4h = df_4h['close'].values
    
    # Calculate EMA(50) on 4h close for trend filter
    ema50_4h = pd.Series(close_4h).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema50_4h_aligned = align_htf_to_ltf(prices, df_4h, ema50_4h)
    
    # Volume filter: current 1h volume > 1.5x 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_filter = volume > (1.5 * vol_ma)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(20, n):  # Start after sufficient data for all indicators
        # Skip if any required data is NaN or outside session
        if (np.isnan(r3[i]) or np.isnan(s3[i]) or 
            np.isnan(ema50_4h_aligned[i]) or np.isnan(vol_ma[i]) or
            not in_session[i]):
            signals[i] = 0.0
            continue
        
        if position == 0:
            # LONG: Price breaks above R3 AND 4h EMA50 uptrend AND volume confirmation
            if close[i] > r3[i] and close[i] > ema50_4h_aligned[i] and volume_filter[i]:
                signals[i] = 0.20
                position = 1
            # SHORT: Price breaks below S3 AND 4h EMA50 downtrend AND volume confirmation
            elif close[i] < s3[i] and close[i] < ema50_4h_aligned[i] and volume_filter[i]:
                signals[i] = -0.20
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: Price returns to Pivot OR 4h EMA50 turns down
            if close[i] <= pivot[i] or close[i] < ema50_4h_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.20
        elif position == -1:
            # EXIT SHORT: Price returns to Pivot OR 4h EMA50 turns up
            if close[i] >= pivot[i] or close[i] > ema50_4h_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.20
    
    return signals
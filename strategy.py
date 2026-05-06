#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h strategy combining weekly pivot points with price action rejection signals
# Uses weekly pivot levels (from prior week) as key support/resistance. Enter long when price
# rejects below S1 with bullish engulfing candle, enter short when price rejects above R1 with
# bearish engulfing candle. Volume confirmation ensures institutional participation. Works in
# both bull and bear markets: rejections at key levels often precede reversals or continuations.
# Target: 50-150 total trades over 4 years (12-37/year) with 0.25 position sizing.

name = "6h_weeklyPivot_Rejection_Volume_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Calculate weekly pivot points ONCE before loop
    df_1w = get_htf_data(prices, '1w')
    
    if len(df_1w) < 2:
        return np.zeros(n)
    
    # Previous week's OHLC for pivot calculation
    prev_close = df_1w['close'].shift(1).values
    prev_high = df_1w['high'].shift(1).values
    prev_low = df_1w['low'].shift(1).values
    
    # Pivot point calculation
    pivot = (prev_high + prev_low + prev_close) / 3
    range_ = prev_high - prev_low
    
    # Support and Resistance levels
    r1 = pivot + range_
    s1 = pivot - range_
    
    # Align weekly levels to 6h timeframe
    r1_aligned = align_htf_to_ltf(prices, df_1w, r1)
    s1_aligned = align_htf_to_ltf(prices, df_1w, s1)
    
    # Bullish engulfing: current green candle engulfs previous red candle
    bullish_engulf = (close > open_) & (open_ < close_prev) & (close >= close_prev) & (open_ <= open_prev)
    # Bearish engulfing: current red candle engulfs previous green candle
    bearish_engulf = (close < open_) & (open_ > close_prev) & (close <= close_prev) & (open_ >= open_prev)
    # Where open_ = open prices, close_prev = previous close, open_prev = previous open
    open_ = prices['open'].values
    close_prev = np.roll(close, 1)
    close_prev[0] = close[0]
    open_prev = np.roll(open_, 1)
    open_prev[0] = open_[0]
    bullish_engulf = (close > open_) & (open_ < close_prev) & (close >= close_prev) & (open_ <= open_prev)
    bearish_engulf = (close < open_) & (open_ > close_prev) & (close <= close_prev) & (open_ >= open_prev)
    
    # Volume confirmation: >1.8x 20-period average
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_filter = volume > (1.8 * vol_ma_20)
    
    # Pre-compute session filter (08-20 UTC)
    hours = pd.DatetimeIndex(prices["open_time"]).hour
    session_filter = (hours >= 8) & (hours <= 20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(50, n):
        # Skip if any critical value is NaN or outside session
        if (np.isnan(r1_aligned[i]) or np.isnan(s1_aligned[i]) or 
            np.isnan(volume_filter[i]) or np.isnan(bullish_engulf[i]) or np.isnan(bearish_engulf[i]) or
            not session_filter[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: bullish engulfing at or above S1 with volume confirmation
            if bullish_engulf[i] and close[i] >= s1_aligned[i] and volume_filter[i]:
                signals[i] = 0.25
                position = 1
            # Short: bearish engulfing at or below R1 with volume confirmation
            elif bearish_engulf[i] and close[i] <= r1_aligned[i] and volume_filter[i]:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: bearish engulfing at or below R1 or price closes below S1
            if bearish_engulf[i] and close[i] <= r1_aligned[i] or close[i] < s1_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: bullish engulfing at or above S1 or price closes above R1
            if bullish_engulf[i] and close[i] >= s1_aligned[i] or close[i] > r1_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals
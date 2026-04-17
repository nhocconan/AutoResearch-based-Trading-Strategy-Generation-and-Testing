#!/usr/bin/env python3
"""
4h_Camarilla_Pivot_R3_S3_Bounce_With_Volume
Hypothesis: Camarilla R3/S3 levels act as strong support/resistance where price often bounces. 
Long when price touches S3 with rejection (close > open) and volume > 1.5x average.
Short when price touches R3 with rejection (close < open) and volume > 1.5x average.
Uses 12h EMA50 as trend filter to avoid counter-trend trades in strong trends.
Designed to work in both bull (buying dips at S3) and bear (selling rallies at R3) markets.
"""

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
    open_ = prices['open'].values
    volume = prices['volume'].values
    
    # Calculate Camarilla levels from previous day
    # R3 = close + (high - low) * 1.1/2
    # S3 = close - (high - low) * 1.1/2
    # We need previous day's OHLC
    prev_day_high = np.roll(high, 1)  # previous bar's high
    prev_day_low = np.roll(low, 1)    # previous bar's low
    prev_day_close = np.roll(close, 1) # previous bar's close
    
    # For first bar, use current values (will be filtered out by warmup anyway)
    prev_day_high[0] = high[0]
    prev_day_low[0] = low[0]
    prev_day_close[0] = close[0]
    
    camarilla_width = (prev_day_high - prev_day_low) * 1.1 / 2
    r3 = prev_day_close + camarilla_width
    s3 = prev_day_close - camarilla_width
    
    # Price rejection at S3/R3: close > open for bullish bounce, close < open for bearish rejection
    bullish_bounce = (close <= s3 * 1.001) & (close > open_)  # Allow small slippage
    bearish_rejection = (close >= r3 * 0.999) & (close < open_)  # Allow small slippage
    
    # Volume confirmation: current volume > 1.5x 20-period average
    volume_series = pd.Series(volume)
    volume_ma20 = volume_series.rolling(window=20, min_periods=20).mean().values
    volume_filter = volume > (1.5 * volume_ma20)
    
    # Get 12h data for trend filter
    df_12h = get_htf_data(prices, '12h')
    close_12h = df_12h['close'].values
    
    # Calculate 12h EMA50 for trend filter
    close_series_12h = pd.Series(close_12h)
    ema50_12h = close_series_12h.ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Align 12h EMA to 4h timeframe
    ema50_12h_aligned = align_htf_to_ltf(prices, df_12h, ema50_12h)
    
    signals = np.zeros(n)
    position = 0  # -1: short, 0: flat, 1: long
    
    start_idx = max(20, 50)  # volume MA20, EMA50
    
    for i in range(start_idx, n):
        # Skip if any required data is not available
        if (np.isnan(volume_ma20[i]) or 
            np.isnan(ema50_12h_aligned[i])):
            signals[i] = 0.0
            continue
        
        vol_filter = volume_filter[i]
        
        if position == 0:
            # Long: price bounces off S3 + volume + 12h uptrend (price > EMA50)
            if bullish_bounce[i] and vol_filter and close[i] > ema50_12h_aligned[i]:
                signals[i] = 0.25
                position = 1
            # Short: price rejected at R3 + volume + 12h downtrend (price < EMA50)
            elif bearish_rejection[i] and vol_filter and close[i] < ema50_12h_aligned[i]:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit long: price reaches R3 or reversal signal
            if close[i] >= r3[i] * 0.999:  # Reached R3 level
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: price reaches S3 or reversal signal
            if close[i] <= s3[i] * 1.001:  # Reached S3 level
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "4h_Camarilla_Pivot_R3_S3_Bounce_With_Volume"
timeframe = "4h"
leverage = 1.0
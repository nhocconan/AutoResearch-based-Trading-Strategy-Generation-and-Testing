#!/usr/bin/env python3
# 12h_price_action_reversal_v1
# Hypothesis: 12-hour price action reversals at weekly support/resistance levels with volume confirmation work in both bull and bear markets.
# Uses price rejection at weekly high/low (pin bar) with volume > 1.5x 20-period average.
# Daily trend filter (EMA50) ensures alignment with intermediate trend.
# Target: 12-30 trades/year (48-120 over 4 years) with controlled risk.

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "12h_price_action_reversal_v1"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 30:
        return np.zeros(n)
    
    open_price = prices['open'].values
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Calculate body and wicks for pin bar detection
    body = np.abs(close - open_price)
    upper_wick = high - np.maximum(close, open_price)
    lower_wick = np.minimum(close, open_price) - low
    
    # Volume confirmation: 20-period average
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_threshold = vol_ma_20 * 1.5
    
    # Daily trend filter: EMA50 on daily data
    df_1d = get_htf_data(prices, '1d')
    close_1d = df_1d['close'].values
    ema_50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # Weekly support/resistance: weekly high/low
    df_1w = get_htf_data(prices, '1w')
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    weekly_high_aligned = align_htf_to_ltf(prices, df_1w, high_1w)
    weekly_low_aligned = align_htf_to_ltf(prices, df_1w, low_1w)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(20, n):
        # Skip if any required data is NaN
        if (np.isnan(body[i]) or np.isnan(upper_wick[i]) or np.isnan(lower_wick[i]) or
            np.isnan(vol_ma_20[i]) or np.isnan(ema_50_1d_aligned[i]) or
            np.isnan(weekly_high_aligned[i]) or np.isnan(weekly_low_aligned[i])):
            signals[i] = 0.0
            continue
        
        if position == 1:  # Long position
            # Exit: price closes below weekly low or daily trend turns bearish
            if close[i] < weekly_low_aligned[i] or close[i] < ema_50_1d_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit: price closes above weekly high or daily trend turns bullish
            if close[i] > weekly_high_aligned[i] or close[i] > ema_50_1d_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat
            # Enter long: bullish pin bar at weekly support with volume confirmation and daily uptrend
            bullish_pin = (lower_wick[i] > 2 * body[i]) and (body[i] > 0)
            at_support = low[i] <= weekly_low_aligned[i] * 1.002  # within 0.2% of weekly low
            if bullish_pin and at_support and volume[i] > vol_threshold[i] and close[i] > ema_50_1d_aligned[i]:
                position = 1
                signals[i] = 0.25
            # Enter short: bearish pin bar at weekly resistance with volume confirmation and daily downtrend
            elif (upper_wick[i] > 2 * body[i]) and (body[i] > 0):  # bearish pin
                at_resistance = high[i] >= weekly_high_aligned[i] * 0.998  # within 0.2% of weekly high
                if at_resistance and volume[i] > vol_threshold[i] and close[i] < ema_50_1d_aligned[i]:
                    position = -1
                    signals[i] = -0.25
    
    return signals
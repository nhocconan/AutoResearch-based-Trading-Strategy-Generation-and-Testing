#!/usr/bin/env python3
# 6h_RSI_Divergence_Volume_Confirmation
# Hypothesis: Combines RSI divergence (bullish/bearish) with volume confirmation and 1d trend filter.
# Bullish divergence: price makes lower low, RSI makes higher low -> long signal when volume confirms.
# Bearish divergence: price makes higher high, RSI makes lower high -> short signal when volume confirms.
# Uses 1d EMA50 as trend filter: only take longs in uptrend, shorts in downtrend.
# Works in bull markets by catching pullbacks in uptrends, works in bear markets by catching bounces in downtrends.
# Volume confirmation reduces false signals. Targets 15-30 trades/year.

name = "6h_RSI_Divergence_Volume_Confirmation"
timeframe = "6h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Get 1d data for EMA trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # Price and volume arrays
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # --- 1d EMA50 for trend filter ---
    close_1d = df_1d['close'].values
    ema_50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1d_slope = ema_50_1d - np.roll(ema_50_1d, 1)
    ema_50_1d_slope[0] = 0
    ema_50_1d_slope = pd.Series(ema_50_1d_slope).ewm(span=3, adjust=False, min_periods=1).mean().values
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    ema_50_1d_slope_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d_slope)
    
    # --- RSI(14) calculation ---
    delta = np.diff(close, prepend=close[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    avg_gain = pd.Series(gain).ewm(span=14, adjust=False, min_periods=14).mean().values
    avg_loss = pd.Series(loss).ewm(span=14, adjust=False, min_periods=14).mean().values
    rs = avg_gain / (avg_loss + 1e-10)
    rsi = 100 - (100 / (1 + rs))
    
    # --- Volume confirmation (volume > 20-period average) ---
    vol_ma = pd.Series(volume).ewm(span=20, adjust=False, min_periods=20).mean().values
    vol_surge = volume > vol_ma
    
    # --- RSI divergence detection (lookback 5 bars) ---
    def detect_divergence(price_series, rsi_series, lookback=5):
        bullish_div = np.zeros_like(price_series, dtype=bool)
        bearish_div = np.zeros_like(price_series, dtype=bool)
        
        for i in range(lookback, len(price_series)):
            # Bullish divergence: price makes lower low, RSI makes higher low
            if (price_series[i] < price_series[i-lookback:i].min() and 
                rsi_series[i] > rsi_series[i-lookback:i].min()):
                bullish_div[i] = True
            # Bearish divergence: price makes higher high, RSI makes lower high
            if (price_series[i] > price_series[i-lookback:i].max() and 
                rsi_series[i] < rsi_series[i-lookback:i].max()):
                bearish_div[i] = True
        return bullish_div, bearish_div
    
    bullish_div, bearish_div = detect_divergence(close, rsi, 5)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Warmup: enough for RSI (14) and EMA50 slope (50+3)
    start_idx = 60
    
    for i in range(start_idx, n):
        # Skip if any critical values are NaN
        if (np.isnan(ema_50_1d_aligned[i]) or
            np.isnan(ema_50_1d_slope_aligned[i]) or
            np.isnan(rsi[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Trend direction from 1d EMA50 slope
        uptrend = ema_50_1d_slope_aligned[i] > 0
        downtrend = ema_50_1d_slope_aligned[i] < 0
        
        if position == 0:
            # Long: bullish divergence + volume surge + uptrend filter
            if bullish_div[i] and vol_surge[i] and uptrend:
                signals[i] = 0.25
                position = 1
            # Short: bearish divergence + volume surge + downtrend filter
            elif bearish_div[i] and vol_surge[i] and downtrend:
                signals[i] = -0.25
                position = -1
        else:
            if position == 1:
                # Exit long: bearish divergence OR trend turns down
                if bearish_div[i] or downtrend:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = 0.25
            elif position == -1:
                # Exit short: bullish divergence OR trend turns up
                if bullish_div[i] or uptrend:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = -0.25
    
    return signals
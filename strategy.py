#!/usr/bin/env python3
"""
6h_1d_OrderBlock_Bounce
Hypothesis: On the 6-hour timeframe, price often respects institutional order blocks derived from 1-day candles.
- Identify bullish order blocks (OB) as the last down candle before a strong up move on the daily chart.
- Identify bearish order blocks as the last up candle before a strong down move on the daily chart.
- On the 6-hour chart, go long when price retraces to a bullish OB and shows rejection (close > open).
- Go short when price retraces to a bearish OB and shows rejection (close < open).
- Use the 1-week trend (price > 1w SMA50) as a filter to only take longs in uptrend and shorts in downtrend.
- This strategy works in bull markets by buying dips at demand zones and in bear markets by selling rallies at supply zones.
- Target: 20-40 trades/year (80-160 total over 4 years) to minimize fee drag.
"""

name = "6h_1d_OrderBlock_Bounce"
timeframe = "6h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Get 1d data for order block detection
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    
    # Get 1w data for trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 50:
        return np.zeros(n)
    
    # 6h OHLCV
    close = prices['close'].values
    open_ = prices['open'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # --- 1d Bullish and Bearish Order Blocks ---
    # Bullish OB: last down candle before a strong up move (close > open and next candle closes above its high)
    # Bearish OB: last up candle before a strong down move (close < open and next candle closes below its low)
    open_1d = df_1d['open'].values
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    bullish_ob_top = np.full(len(close_1d), np.nan)   # high of the OB candle
    bullish_ob_bottom = np.full(len(close_1d), np.nan) # low of the OB candle
    bearish_ob_top = np.full(len(close_1d), np.nan)    # high of the OB candle
    bearish_ob_bottom = np.full(len(close_1d), np.nan) # low of the OB candle
    
    # Detect bullish OBs
    for i in range(1, len(close_1d)-1):
        # Current candle is down
        if close_1d[i] < open_1d[i]:
            # Next candle is strong up (closes above current candle's high)
            if close_1d[i+1] > high_1d[i]:
                bullish_ob_top[i] = high_1d[i]
                bullish_ob_bottom[i] = low_1d[i]
    
    # Detect bearish OBs
    for i in range(1, len(close_1d)-1):
        # Current candle is up
        if close_1d[i] > open_1d[i]:
            # Next candle is strong down (closes below current candle's low)
            if close_1d[i+1] < low_1d[i]:
                bearish_ob_top[i] = high_1d[i]
                bearish_ob_bottom[i] = low_1d[i]
    
    # --- 1w Trend Filter (price > 1w SMA50) ---
    close_1w = df_1w['close'].values
    sma_1w = np.full(len(close_1w), np.nan)
    for i in range(50, len(close_1w)):
        sma_1w[i] = np.mean(close_1w[i-50:i])
    sma_1w_aligned = align_htf_to_ltf(prices, df_1w, sma_1w)
    
    # --- Align OB levels to 6h ---
    bullish_ob_top_aligned = align_htf_to_ltf(prices, df_1d, bullish_ob_top)
    bullish_ob_bottom_aligned = align_htf_to_ltf(prices, df_1d, bullish_ob_bottom)
    bearish_ob_top_aligned = align_htf_to_ltf(prices, df_1d, bearish_ob_top)
    bearish_ob_bottom_aligned = align_htf_to_ltf(prices, df_1d, bearish_ob_bottom)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Warmup: need 1d and 1w data
    start_idx = 50  # cover 1w SMA50
    
    for i in range(start_idx, n):
        # Skip if trend filter not ready
        if np.isnan(sma_1w_aligned[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Determine 1w trend
        trend_up = close[i] > sma_1w_aligned[i]   # price above 1w SMA50
        trend_down = close[i] < sma_1w_aligned[i] # price below 1w SMA50
        
        # Check if current price is within any bullish OB
        in_bullish_ob = False
        if not (np.isnan(bullish_ob_top_aligned[i]) or np.isnan(bullish_ob_bottom_aligned[i])):
            ob_high = bullish_ob_top_aligned[i]
            ob_low = bullish_ob_bottom_aligned[i]
            if ob_low <= close[i] <= ob_high:
                in_bullish_ob = True
        
        # Check if current price is within any bearish OB
        in_bearish_ob = False
        if not (np.isnan(bearish_ob_top_aligned[i]) or np.isnan(bearish_ob_bottom_aligned[i])):
            ob_high = bearish_ob_top_aligned[i]
            ob_low = bearish_ob_bottom_aligned[i]
            if ob_low <= close[i] <= ob_high:
                in_bearish_ob = True
        
        # Rejection candle: close > open for bullish, close < open for bearish
        bullish_rejection = close[i] > open_[i]
        bearish_rejection = close[i] < open_[i]
        
        if position == 0:
            # Long: price in bullish OB, bullish rejection, and 1w uptrend
            if in_bullish_ob and bullish_rejection and trend_up:
                signals[i] = 0.25
                position = 1
            # Short: price in bearish OB, bearish rejection, and 1w downtrend
            elif in_bearish_ob and bearish_rejection and trend_down:
                signals[i] = -0.25
                position = -1
        else:
            if position == 1:
                # Exit long: price closes below OB low or 1w trend turns down
                if not np.isnan(bullish_ob_bottom_aligned[i]) and close[i] < bullish_ob_bottom_aligned[i]:
                    signals[i] = 0.0
                    position = 0
                elif not trend_up:  # 1w trend turned down
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = 0.25
            elif position == -1:
                # Exit short: price closes above OB high or 1w trend turns up
                if not np.isnan(bearish_ob_top_aligned[i]) and close[i] > bearish_ob_top_aligned[i]:
                    signals[i] = 0.0
                    position = 0
                elif not trend_down:  # 1w trend turned up
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = -0.25
    
    return signals
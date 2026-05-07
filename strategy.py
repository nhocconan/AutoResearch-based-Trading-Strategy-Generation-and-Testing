#!/usr/bin/env python3
# 4h_RSI_Trend_Squeeze_Bounce
# Hypothesis: Combines RSI mean reversion with trend alignment and volatility squeeze to capture high-probability bounces in both bull and bear markets.
# Uses 1-day RSI (14) for mean reversion signals (oversold/overbought), aligned with 4-hour trend (EMA50) and Bollinger Band squeeze (low volatility) for entry.
# Exits on RSI mean reversion or trend violation. Designed for low trade frequency (20-40/year) to minimize fee drag while maintaining edge in choppy and trending regimes.
# Works in bull markets by buying oversold dips in uptrends and in bear markets by selling overbought rallies in downtrends.

name = "4h_RSI_Trend_Squeeze_Bounce"
timeframe = "4h"
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
    
    # Get 1-day data for RSI and Bollinger Bands
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # Calculate 1-day RSI (14-period)
    close_1d = df_1d['close'].values
    delta = np.diff(close_1d, prepend=close_1d[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    
    # Wilder's smoothing (equivalent to EMA with alpha=1/14)
    avg_gain = np.zeros_like(gain)
    avg_loss = np.zeros_like(loss)
    avg_gain[13] = np.mean(gain[1:14])  # First average of first 14 gains
    avg_loss[13] = np.mean(loss[1:14])  # First average of first 14 losses
    
    for i in range(14, len(gain)):
        avg_gain[i] = (avg_gain[i-1] * 13 + gain[i]) / 14
        avg_loss[i] = (avg_loss[i-1] * 13 + loss[i]) / 14
    
    rs = np.divide(avg_gain, avg_loss, out=np.zeros_like(avg_gain), where=avg_loss!=0)
    rsi_1d = 100 - (100 / (1 + rs))
    rsi_1d[:14] = np.nan  # Not enough data for first 14 periods
    
    # Calculate Bollinger Bands (20, 2) on 1-day
    sma_20 = pd.Series(close_1d).rolling(window=20, min_periods=20).mean().values
    std_20 = pd.Series(close_1d).rolling(window=20, min_periods=20).std().values
    upper_bb = sma_20 + 2 * std_20
    lower_bb = sma_20 - 2 * std_20
    bb_width = (upper_bb - lower_bb) / sma_20  # Normalized width
    
    # Align 1D indicators to 4h
    rsi_1d_4h = align_htf_to_ltf(prices, df_1d, rsi_1d)
    bb_width_4h = align_htf_to_ltf(prices, df_1d, bb_width)
    
    # Calculate 4-hour EMA50 for trend filter
    ema_50 = pd.Series(close).ewm(span=50, adjust=False, min_periods=50).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(50, n):  # Start after EMA50 warmup
        # Skip if any critical value is NaN
        if (np.isnan(rsi_1d_4h[i]) or np.isnan(bb_width_4h[i]) or np.isnan(ema_50[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: RSI oversold (<30) + low volatility squeeze (BB width < 0.05) + price above EMA50 (uptrend)
            if rsi_1d_4h[i] < 30 and bb_width_4h[i] < 0.05 and close[i] > ema_50[i]:
                signals[i] = 0.25
                position = 1
            # Short: RSI overbought (>70) + low volatility squeeze (BB width < 0.05) + price below EMA50 (downtrend)
            elif rsi_1d_4h[i] > 70 and bb_width_4h[i] < 0.05 and close[i] < ema_50[i]:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit: RSI mean reversion (>50) or price breaks below EMA50 (trend failure)
            if rsi_1d_4h[i] > 50 or close[i] < ema_50[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit: RSI mean reversion (<50) or price breaks above EMA50 (trend failure)
            if rsi_1d_4h[i] < 50 or close[i] > ema_50[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals
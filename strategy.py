#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1h strategy using 4h RSI and 1d EMA50 for trend direction, with 1h price action for entry timing.
# Long when 4h RSI > 50 (bullish) and 1h close > 1d EMA50 (above long-term trend) with bullish engulfing candle.
# Short when 4h RSI < 50 (bearish) and 1h close < 1d EMA50 (below long-term trend) with bearish engulfing candle.
# Uses 4h RSI for medium-term trend filter, 1d EMA50 for long-term trend filter, and 1h price action for precise entry.
# Designed to reduce false signals in ranging markets and capture momentum in trending markets.
# Target: 15-30 trades per year (60-120 over 4 years) with 0.20 position sizing.
# Session filter: 08-20 UTC to avoid low-volume Asian session.

name = "1h_4hRSI_1dEMA50_Engulfing_v1"
timeframe = "1h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    open_price = prices['open'].values
    high = prices['high'].values
    low = prices['low'].values
    
    # Get 4h data for RSI
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 14:
        return np.zeros(n)
    
    # Calculate 4h RSI (14-period)
    delta = pd.Series(df_4h['close']).diff()
    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)
    avg_gain = gain.ewm(alpha=1/14, adjust=False, min_periods=14).mean()
    avg_loss = loss.ewm(alpha=1/14, adjust=False, min_periods=14).mean()
    rs = avg_gain / avg_loss
    rsi_4h = 100 - (100 / (1 + rs))
    rsi_4h_values = rsi_4h.values
    rsi_4h_aligned = align_htf_to_ltf(prices, df_4h, rsi_4h_values)
    
    # Get 1d data for EMA50
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # Calculate 1d EMA50
    ema_50_1d = pd.Series(df_1d['close']).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # Pre-compute session filter (08-20 UTC)
    hours = pd.DatetimeIndex(prices["open_time"]).hour
    session_filter = (hours >= 8) & (hours <= 20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(1, n):  # Start from 1 to access previous candle for engulfing
        # Skip if any critical value is NaN or outside session
        if (np.isnan(rsi_4h_aligned[i]) or np.isnan(ema_50_1d_aligned[i]) or
            not session_filter[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Bullish engulfing: current green candle completely engulfs previous red candle
        bullish_engulfing = (close[i] > open_price[i]) and (open_price[i] < close[i-1]) and (close[i] > open_price[i-1])
        # Bearish engulfing: current red candle completely engulfs previous green candle
        bearish_engulfing = (close[i] < open_price[i]) and (open_price[i] > close[i-1]) and (close[i] < open_price[i-1])
        
        if position == 0:
            # Long: bullish 4h RSI, above 1d EMA50, and bullish engulfing
            if (rsi_4h_aligned[i] > 50) and (close[i] > ema_50_1d_aligned[i]) and bullish_engulfing:
                signals[i] = 0.20
                position = 1
            # Short: bearish 4h RSI, below 1d EMA50, and bearish engulfing
            elif (rsi_4h_aligned[i] < 50) and (close[i] < ema_50_1d_aligned[i]) and bearish_engulfing:
                signals[i] = -0.20
                position = -1
        elif position == 1:
            # Exit long: bearish engulfing or price crosses below 1d EMA50
            if bearish_engulfing or (close[i] < ema_50_1d_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.20
        elif position == -1:
            # Exit short: bullish engulfing or price crosses above 1d EMA50
            if bullish_engulfing or (close[i] > ema_50_1d_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.20
    
    return signals
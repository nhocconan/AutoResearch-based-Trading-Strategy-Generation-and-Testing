#!/usr/bin/env python3
# 1d_1w_RSI_Divergence_TrendFilter_v1
# Hypothesis: On 1d timeframe, trade weekly RSI divergence signals with trend filter.
# Weekly RSI divergence identifies potential reversals, while daily trend filter (EMA50) 
# ensures we trade with the intermediate trend. This combination works in both bull and bear markets
# by capturing mean reversion within the trend context. Targets 15-25 trades/year with strict entry conditions.

name = "1d_1w_RSI_Divergence_TrendFilter_v1"
timeframe = "1d"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def calculate_rsi(close_prices, period=14):
    """Calculate RSI with proper Wilder smoothing"""
    if len(close_prices) < period + 1:
        return np.full_like(close_prices, np.nan)
    
    delta = np.diff(close_prices)
    gain = np.where(delta > 0, delta, 0.0)
    loss = np.where(delta < 0, -delta, 0.0)
    
    # First average gain/loss
    avg_gain = np.zeros_like(close_prices)
    avg_loss = np.zeros_like(close_prices)
    avg_gain[period] = np.mean(gain[:period])
    avg_loss[period] = np.mean(loss[:period])
    
    # Wilder smoothing
    for i in range(period + 1, len(close_prices)):
        avg_gain[i] = (avg_gain[i-1] * (period - 1) + gain[i-1]) / period
        avg_loss[i] = (avg_loss[i-1] * (period - 1) + loss[i-1]) / period
    
    rs = np.where(avg_loss != 0, avg_gain / avg_loss, 0)
    rsi = 100 - (100 / (1 + rs))
    return rsi

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get weekly data ONCE before loop
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 30:
        return np.zeros(n)
    
    # Calculate weekly RSI (14-period)
    close_1w = df_1w['close'].values
    rsi_1w = calculate_rsi(close_1w, 14)
    
    # Align weekly RSI to daily timeframe
    rsi_1w_aligned = align_htf_to_ltf(prices, df_1w, rsi_1w)
    
    # Calculate daily EMA50 for trend filter
    close_series = pd.Series(close)
    ema50 = close_series.ewm(span=50, adjust=False, min_periods=50).values
    
    # Calculate volume average for confirmation
    volume_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 60  # Ensure indicators are ready
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(rsi_1w_aligned[i]) or 
            np.isnan(ema50[i]) or 
            np.isnan(volume_ma[i])):
            signals[i] = 0.0
            continue
        
        if position == 0:
            # Look for RSI divergence signals
            bullish_divergence = False
            bearish_divergence = False
            
            # Check for bullish divergence: price making lower low, RSI making higher low
            if i >= 20:  # Need sufficient lookback for divergence
                # Find recent swing lows in price and RSI
                price_low_idx = i - np.argmin(low[i-20:i+1])  # index of lowest price in last 20 periods
                rsi_low_idx = i - np.argmin(rsi_1w_aligned[i-20:i+1])  # index of lowest RSI in last 20 periods
                
                # Bullish divergence: price lower low, RSI higher low
                if (price_low_idx != rsi_low_idx and 
                    low[i] < low[price_low_idx] and  # price made lower low
                    rsi_1w_aligned[i] > rsi_1w_aligned[rsi_low_idx]):  # RSI made higher low
                    bullish_divergence = True
                
                # Bearish divergence: price higher high, RSI lower high
                price_high_idx = i - np.argmax(high[i-20:i+1])  # index of highest price in last 20 periods
                rsi_high_idx = i - np.argmax(rsi_1w_aligned[i-20:i+1])  # index of highest RSI in last 20 periods
                
                if (price_high_idx != rsi_high_idx and 
                    high[i] > high[price_high_idx] and  # price made higher high
                    rsi_1w_aligned[i] < rsi_1w_aligned[rsi_high_idx]):  # RSI made lower high
                    bearish_divergence = True
            
            # Enter long on bullish divergence with trend filter and volume confirmation
            if bullish_divergence and close[i] > ema50[i] and volume[i] > 1.5 * volume_ma[i]:
                signals[i] = 0.25
                position = 1
            # Enter short on bearish divergence with trend filter and volume confirmation
            elif bearish_divergence and close[i] < ema50[i] and volume[i] > 1.5 * volume_ma[i]:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit long: RSI overbought or trend change
            if rsi_1w_aligned[i] > 70 or close[i] < ema50[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: RSI oversold or trend change
            if rsi_1w_aligned[i] < 30 or close[i] > ema50[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals
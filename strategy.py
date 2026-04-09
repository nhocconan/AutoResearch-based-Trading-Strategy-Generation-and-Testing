#!/usr/bin/env python3
# 1d_1w_rsi_divergence_v1
# Hypothesis: Weekly RSI divergence on daily timeframe with volume confirmation and trend filter.
# Bullish divergence: price makes lower low while RSI makes higher low on weekly chart → long when price closes above prior swing high.
# Bearish divergence: price makes higher high while RSI makes lower high on weekly chart → short when price closes below prior swing low.
# Works in bull markets via momentum continuation and in bear markets via mean reversion at exhaustion points.
# Uses weekly RSI divergence as primary signal, daily EMA50 for trend filter, and volume > 1.5x 20-bar average for confirmation.
# Position size: 0.25 to limit drawdown. Target: 30-100 total trades over 4 years (7-25/year).

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_ltf_to_htf  # Note: corrected import name

def calculate_rsi(prices, period=14):
    """Calculate RSI with proper Wilder's smoothing."""
    delta = np.diff(prices)
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    
    avg_gain = np.zeros_like(prices)
    avg_loss = np.zeros_like(prices)
    
    # Initialize first average
    if len(prices) > period:
        avg_gain[period] = np.mean(gain[:period])
        avg_loss[period] = np.mean(loss[:period])
        
        # Wilder's smoothing
        for i in range(period + 1, len(prices)):
            avg_gain[i] = (avg_gain[i-1] * (period - 1) + gain[i-1]) / period
            avg_loss[i] = (avg_loss[i-1] * (period - 1) + loss[i-1]) / period
    
    rs = np.divide(avg_gain, avg_loss, out=np.full_like(avg_gain, np.nan), where=avg_loss!=0)
    rsi = 100 - (100 / (1 + rs))
    return rsi

def find_swing_points(high, low, window=2):
    """Find swing highs and lows."""
    swing_high = np.full_like(high, np.nan)
    swing_low = np.full_like(low, np.nan)
    
    for i in range(window, len(high) - window):
        if high[i] == np.max(high[i-window:i+window+1]):
            swing_high[i] = high[i]
        if low[i] == np.min(low[i-window:i+window+1]):
            swing_low[i] = low[i]
    
    return swing_high, swing_low

def detect_divergence(price, indicator, lookback=5):
    """Detect bullish/bearish divergence."""
    n = len(price)
    bullish_div = np.zeros(n, dtype=bool)
    bearish_div = np.zeros(n, dtype=bool)
    
    for i in range(lookback, n):
        # Look for lower low in price, higher low in indicator (bullish div)
        if (price[i] < price[i-lookback] and 
            indicator[i] > indicator[i-lookback] and
            np.all(price[i-lookback+1:i] >= price[i-lookback]) and
            np.all(indicator[i-lookback+1:i] <= indicator[i-lookback])):
            bullish_div[i] = True
            
        # Look for higher high in price, lower high in indicator (bearish div)
        if (price[i] > price[i-lookback] and 
            indicator[i] < indicator[i-lookback] and
            np.all(price[i-lookback+1:i] <= price[i-lookback]) and
            np.all(indicator[i-lookback+1:i] >= indicator[i-lookback])):
            bearish_div[i] = True
    
    return bullish_div, bearish_div

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Load weekly data ONCE before loop
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 30:
        return np.zeros(n)
    
    # Calculate weekly RSI (14)
    close_1w = df_1w['close'].values
    rsi_1w = calculate_rsi(close_1w, 14)
    
    # Align weekly RSI to daily timeframe
    rsi_1w_aligned = align_ltf_to_htf(prices, df_1w, rsi_1w)
    
    # Calculate daily EMA50 for trend filter
    ema_50 = np.full(n, np.nan)
    if n >= 50:
        ema = np.mean(close[:50])
        ema_50[49] = ema
        multiplier = 2 / (50 + 1)
        for i in range(50, n):
            ema = (close[i] - ema) * multiplier + ema
            ema_50[i] = ema
    
    # Volume confirmation: 20-period average
    vol_ma_20 = np.full(n, np.nan)
    vol_sum = 0
    for i in range(n):
        vol_sum += volume[i]
        if i >= 20:
            vol_sum -= volume[i-20]
        if i >= 19:
            vol_ma_20[i] = vol_sum / 20
    
    # Detect swing points on weekly data for divergence confirmation
    swing_high_1w, swing_low_1w = find_swing_points(
        df_1w['high'].values, 
        df_1w['low'].values, 
        window=2
    )
    
    # Align swing points to daily timeframe
    swing_high_1w_aligned = align_ltf_to_htf(prices, df_1w, swing_high_1w)
    swing_low_1w_aligned = align_ltf_to_htf(prices, df_1w, swing_low_1w)
    
    # Detect RSI divergence on weekly data
    bullish_div, bearish_div = detect_divergence(close_1w, rsi_1w, lookback=5)
    
    # Align divergence signals to daily timeframe
    bullish_div_aligned = align_ltf_to_htf(prices, df_1w, bullish_div.astype(float))
    bearish_div_aligned = align_ltf_to_htf(prices, df_1w, bearish_div.astype(float))
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(50, n):
        # Skip if any required data is invalid
        if (np.isnan(rsi_1w_aligned[i]) or 
            np.isnan(ema_50[i]) or 
            np.isnan(vol_ma_20[i])):
            signals[i] = 0.0
            continue
        
        if position == 1:  # Long position
            # Exit: RSI shows bearish divergence or price closes below EMA50
            if (bearish_div_aligned[i] > 0.5 or close[i] < ema_50[i]):
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit: RSI shows bullish divergence or price closes above EMA50
            if (bullish_div_aligned[i] > 0.5 or close[i] > ema_50[i]):
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat
            # Enter long: bullish divergence on weekly RSI with trend and volume filters
            if (bullish_div_aligned[i] > 0.5 and 
                close[i] > ema_50[i] and 
                volume[i] > vol_ma_20[i] * 1.5):
                position = 1
                signals[i] = 0.25
            # Enter short: bearish divergence on weekly RSI with trend and volume filters
            elif (bearish_div_aligned[i] > 0.5 and 
                  close[i] < ema_50[i] and 
                  volume[i] > vol_ma_20[i] * 1.5):
                position = -1
                signals[i] = -0.25
    
    return signals

name = "1d_1w_rsi_divergence_v1"
timeframe = "1d"
leverage = 1.0
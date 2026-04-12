#!/usr/bin/env python3
"""
12h_1d_RSI_Divergence_v1
Hypothesis: Trade RSI divergences on 12h timeframe with 1-day trend filter. 
In bull markets: buy bullish RSI divergence (price makes lower low, RSI makes higher low) when 1-day trend is up.
In bear markets: sell bearish RSI divergence (price makes higher high, RSI makes lower high) when 1-day trend is down.
Uses volume confirmation to avoid false signals. Designed for low-frequency, high-conviction trades (15-25/year) that work in both bull and bear regimes by trading against short-term momentum within the longer-term trend.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "12h_1d_RSI_Divergence_v1"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # === 1-DAY DATA FOR TREND FILTER ===
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    
    # 1-day EMA50 for trend filter
    ema50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    trend_up_1d = ema50_1d > np.roll(ema50_1d, 1)
    trend_down_1d = ema50_1d < np.roll(ema50_1d, 1)
    trend_up_1d[0] = False
    trend_down_1d[0] = False
    
    # Align 1-day trend to 12h
    trend_up_aligned = align_htf_to_ltf(prices, df_1d, trend_up_1d)
    trend_down_aligned = align_htf_to_ltf(prices, df_1d, trend_down_1d)
    
    # === 12H INDICATORS: RSI AND VOLUME ===
    # RSI(14)
    rsi_period = 14
    delta = np.diff(close, prepend=close[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    avg_gain = pd.Series(gain).ewm(span=rsi_period, adjust=False, min_periods=rsi_period).mean().values
    avg_loss = pd.Series(loss).ewm(span=rsi_period, adjust=False, min_periods=rsi_period).mean().values
    rs = np.where(avg_loss != 0, avg_gain / avg_loss, 0)
    rsi = 100 - (100 / (1 + rs))
    
    # Volume average (20-period)
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    # Lookback period for divergence detection
    lookback = 10
    
    for i in range(lookback, n):
        # Skip if not ready
        if (np.isnan(rsi[i]) or np.isnan(vol_ma[i]) or 
            np.isnan(trend_up_aligned[i]) or np.isnan(trend_down_aligned[i])):
            signals[i] = 0.0 if position == 0 else (0.25 if position == 1 else -0.25)
            continue
        
        # Volume confirmation
        strong_volume = volume[i] > (vol_ma[i] * 1.5)
        
        # Initialize divergence flags
        bullish_div = False
        bearish_div = False
        
        # Look for bullish divergence: price makes lower low, RSI makes higher low
        if i >= lookback:
            # Find local low in price over lookback period
            price_low_idx = i - np.argmin(low[i-lookback:i+1])
            rsi_low_idx = i - np.argmin(rsi[i-lookback:i+1])
            
            # Bullish divergence: price lower low but RSI higher low
            if (price_low_idx == i and  # current bar is price low
                low[i] < low[price_low_idx] and  # made new low in price
                rsi[i] > rsi[rsi_low_idx] and   # but RSI made higher low
                rsi_low_idx < price_low_idx):   # RSI low occurred before price low
                bullish_div = True
        
        # Look for bearish divergence: price makes higher high, RSI makes lower high
        if i >= lookback:
            # Find local high in price over lookback period
            price_high_idx = i - np.argmax(high[i-lookback:i+1])
            rsi_high_idx = i - np.argmax(rsi[i-lookback:i+1])
            
            # Bearish divergence: price higher high but RSI lower high
            if (price_high_idx == i and  # current bar is price high
                high[i] > high[price_high_idx] and  # made new high in price
                rsi[i] < rsi[rsi_high_idx] and      # but RSI made lower high
                rsi_high_idx < price_high_idx):     # RSI high occurred before price high
                bearish_div = True
        
        # Long: bullish divergence + uptrend + volume
        long_signal = bullish_div and trend_up_aligned[i] and strong_volume
        
        # Short: bearish divergence + downtrend + volume
        short_signal = bearish_div and trend_down_aligned[i] and strong_volume
        
        # Exit: opposite divergence or trend weakness
        exit_long = position == 1 and (bearish_div or not trend_up_aligned[i])
        exit_short = position == -1 and (bullish_div or not trend_down_aligned[i])
        
        # Execute trades
        if long_signal and position != 1:
            position = 1
            signals[i] = 0.25
        elif short_signal and position != -1:
            position = -1
            signals[i] = -0.25
        elif exit_long and position == 1:
            position = 0
            signals[i] = 0.0
        elif exit_short and position == -1:
            position = 0
            signals[i] = 0.0
        else:
            # Hold position
            signals[i] = 0.25 if position == 1 else (-0.25 if position == -1 else 0.0)
    
    return signals
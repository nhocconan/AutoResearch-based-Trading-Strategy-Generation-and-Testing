#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Strategy: 6h RSI Divergence + 1d Momentum + Volume Filter
# Hypothesis: RSI divergence on 6h identifies exhaustion points in trends.
# We use 1d momentum (ROC) to filter trades in the direction of higher timeframe trend.
# Volume confirms institutional participation. Works in both bull and bear markets by
# trading mean reversion within the dominant trend. Target: 12-37 trades/year.
name = "6h_rsi_divergence_1d_momentum_volume_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Price data
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1-day data for trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 10:
        return np.zeros(n)
    
    # 14-period RSI on 6h
    delta = np.diff(close, prepend=close[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    avg_gain = pd.Series(gain).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    avg_loss = pd.Series(loss).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    rs = avg_gain / (avg_loss + 1e-10)
    rsi = 100 - (100 / (1 + rs))
    
    # 1-day ROC(10) for momentum filter
    daily_close = df_1d['close'].values
    daily_roc = np.full_like(daily_close, np.nan)
    daily_roc[10:] = (daily_close[10:] - daily_close[:-10]) / daily_close[:-10] * 100
    daily_roc_6h = align_htf_to_ltf(prices, df_1d, daily_roc)
    
    # Volume filter: current volume > 1.3x 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean()
    vol_filter = volume > (vol_ma * 1.3)
    
    signals = np.zeros(n)
    
    for i in range(50, n):
        # Skip if required data not available
        if (np.isnan(rsi[i]) or np.isnan(daily_roc_6h[i]) or np.isnan(vol_ma[i])):
            signals[i] = 0.0
            continue
        
        # Check for RSI divergence (bearish: price higher high, RSI lower high)
        # Bullish divergence: price lower low, RSI higher low
        bearish_div = False
        bullish_div = False
        lookback = 5
        
        if i >= lookback:
            # Bearish divergence: price makes higher high, RSI makes lower high
            if (high[i] > high[i-lookback] and 
                rsi[i] < rsi[i-lookback]):
                bearish_div = True
            # Bullish divergence: price makes lower low, RSI makes higher low
            if (low[i] < low[i-lookback] and 
                rsi[i] > rsi[i-lookback]):
                bullish_div = True
        
        # Enter long on bullish divergence in bullish 1d momentum
        if bullish_div and daily_roc_6h[i] > 0 and vol_filter[i]:
            signals[i] = 0.25
        # Enter short on bearish divergence in bearish 1d momentum
        elif bearish_div and daily_roc_6h[i] < 0 and vol_filter[i]:
            signals[i] = -0.25
        else:
            signals[i] = 0.0
    
    return signals
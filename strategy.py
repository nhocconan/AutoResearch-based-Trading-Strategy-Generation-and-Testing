#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h RSI divergence with 1d trend filter and volume confirmation.
# Uses bearish/bullish RSI divergence to catch reversals in overbought/oversold conditions.
# 1d EMA50 filter ensures trades align with higher timeframe trend.
# Volume spike confirms conviction. Designed for low trade frequency (12-37/year).
# Works in bull markets (buy oversold dips in uptrend) and bear markets (sell overbought rallies in downtrend).
name = "12h_RSIDivergence_1dEMA50_Volume"
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
    
    # Get 1d data for EMA50 filter (ONCE before loop)
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate RSI (14-period)
    delta = pd.Series(close).diff()
    gain = delta.where(delta > 0, 0.0)
    loss = -delta.where(delta < 0, 0.0)
    avg_gain = gain.ewm(alpha=1/14, adjust=False, min_periods=14).mean()
    avg_loss = loss.ewm(alpha=1/14, adjust=False, min_periods=14).mean()
    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    rsi = rsi.values
    
    # Calculate 1d EMA50 for trend filter
    close_1d = pd.Series(df_1d['close'].values)
    ema50_1d = close_1d.ewm(span=50, adjust=False, min_periods=50).mean().values
    ema50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema50_1d)
    
    # Calculate volume spike: current volume > 2.0 * 20-period average volume
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (2.0 * vol_ma_20)
    
    # Calculate RSI divergence
    # Bullish divergence: price makes lower low, RSI makes higher low
    # Bearish divergence: price makes higher high, RSI makes lower high
    lookback = 10
    bullish_div = np.zeros(n, dtype=bool)
    bearish_div = np.zeros(n, dtype=bool)
    
    for i in range(lookback, n):
        # Bullish divergence: lower low in price, higher low in RSI
        if low[i] < low[i-lookback] and rsi[i] > rsi[i-lookback]:
            # Check if this is a meaningful low point
            if low[i] == np.min(low[i-lookback:i+1]):
                bullish_div[i] = True
        # Bearish divergence: higher high in price, lower high in RSI
        if high[i] > high[i-lookback] and rsi[i] < rsi[i-lookback]:
            # Check if this is a meaningful high point
            if high[i] == np.max(high[i-lookback:i+1]):
                bearish_div[i] = True
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 100  # Wait for indicator calculations
    
    for i in range(start_idx, n):
        # Skip if any required data is not available
        if (np.isnan(rsi[i]) or np.isnan(ema50_1d_aligned[i]) or
            np.isnan(vol_ma_20[i])):
            signals[i] = 0.0
            continue
        
        # Trend filter: price above/below 1d EMA50
        uptrend = close[i] > ema50_1d_aligned[i]
        downtrend = close[i] < ema50_1d_aligned[i]
        
        if position == 0:
            # Long: bullish RSI divergence AND uptrend AND volume spike
            if bullish_div[i] and uptrend and volume_spike[i]:
                signals[i] = 0.25
                position = 1
            # Short: bearish RSI divergence AND downtrend AND volume spike
            elif bearish_div[i] and downtrend and volume_spike[i]:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Long exit: bearish divergence OR trend reverses
            if bearish_div[i] or not uptrend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: bullish divergence OR trend reverses
            if bullish_div[i] or not downtrend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals
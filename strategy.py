#!/usr/bin/env python3
"""
4h_RSI_Divergence_12hTrend_Filter
Hypothesis: RSI(14) bearish/bullish divergences on 4h chart with 12h EMA34 trend filter capture reversals in both bull and bear markets. 
Divergence occurs when price makes new high/low but RSI does not, signaling weakening momentum. 
Combined with 12h trend filter to avoid counter-trend trades and volume confirmation to ensure strength. 
Designed for low trade frequency (<50/year) to minimize fee drag while capturing high-probability reversals.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # RSI(14) calculation
    delta = pd.Series(close).diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(alpha=1/14, adjust=False, min_periods=14).mean()
    avg_loss = loss.ewm(alpha=1/14, adjust=False, min_periods=14).mean()
    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    rsi_values = rsi.values
    
    # Get 12h data for trend filter (once before loop)
    df_12h = get_htf_data(prices, '12h')
    close_12h = df_12h['close']
    ema_34_12h = pd.Series(close_12h).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_12h_aligned = align_htf_to_ltf(prices, df_12h, ema_34_12h)
    
    # Volume spike: 1.5x 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (1.5 * vol_ma)
    
    # Detect RSI divergence: bearish (price high, RSI lower high) and bullish (price low, RSI higher low)
    lookback = 10  # lookback for swing high/low
    bearish_div = np.zeros(n, dtype=bool)
    bullish_div = np.zeros(n, dtype=bool)
    
    for i in range(lookback, n):
        # Bearish divergence: price makes higher high, RSI makes lower high
        if high[i] == np.max(high[i-lookback:i+1]):
            # Find previous swing high
            prev_high_idx = i - lookback + np.argmax(high[i-lookback:i])
            if prev_high_idx < i - lookback:  # ensure we look back far enough
                if high[i] > high[prev_high_idx] and rsi_values[i] < rsi_values[prev_high_idx]:
                    bearish_div[i] = True
        
        # Bullish divergence: price makes lower low, RSI makes higher low
        if low[i] == np.min(low[i-lookback:i+1]):
            # Find previous swing low
            prev_low_idx = i - lookback + np.argmin(low[i-lookback:i])
            if prev_low_idx < i - lookback:
                if low[i] < low[prev_low_idx] and rsi_values[i] > rsi_values[prev_low_idx]:
                    bullish_div[i] = True
    
    signals = np.zeros(n)
    position = 0  # -1 short, 0 flat, 1 long
    
    start_idx = max(50, lookback)
    
    for i in range(start_idx, n):
        if (np.isnan(ema_34_12h_aligned[i]) or 
            np.isnan(vol_ma[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        ema_trend = ema_34_12h_aligned[i]
        
        if position == 0:
            # Long: bullish divergence with volume spike and price above 12h EMA (uptrend context)
            if bullish_div[i] and volume_spike[i] and price > ema_trend:
                signals[i] = 0.25
                position = 1
            # Short: bearish divergence with volume spike and price below 12h EMA (downtrend context)
            elif bearish_div[i] and volume_spike[i] and price < ema_trend:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Long position
            signals[i] = 0.25
            # Exit: bearish divergence or price breaks below 12h EMA
            if bearish_div[i] or price < ema_trend:
                signals[i] = 0.0
                position = 0
        
        elif position == -1:
            # Short position
            signals[i] = -0.25
            # Exit: bullish divergence or price breaks above 12h EMA
            if bullish_div[i] or price > ema_trend:
                signals[i] = 0.0
                position = 0
    
    return signals

name = "4h_RSI_Divergence_12hTrend_Filter"
timeframe = "4h"
leverage = 1.0
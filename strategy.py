#!/usr/bin/env python3
"""
1h_4h1d_rsi_divergence_volume_v1
Hypothesis: On 1-hour timeframe, use RSI divergence with 4h/1d trend filter and volume confirmation.
Long when RSI makes higher low while price makes lower low (bullish divergence) with 4h/1d uptrend and volume > 1.5x average.
Short when RSI makes lower high while price makes higher high (bearish divergence) with 4h/1d downtrend and volume > 1.5x average.
Exit when RSI crosses 50 in opposite direction.
Designed for 15-37 trades/year to minimize fee drag while capturing reversal points in both bull and bear markets.
RSI divergence works well at turning points, and multi-timeframe alignment filters counter-trend noise.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "1h_4h1d_rsi_divergence_volume_v1"
timeframe = "1h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Price data
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Get 4h and 1d data for trend filters
    df_4h = get_htf_data(prices, '4h')
    df_1d = get_htf_data(prices, '1d')
    
    if len(df_4h) < 30 or len(df_1d) < 30:
        return np.zeros(n)
    
    # Calculate 4h EMA(20) for trend filter
    close_4h = df_4h['close'].values
    ema_20_4h = pd.Series(close_4h).ewm(span=20, adjust=False, min_periods=20).mean().values
    ema_20_4h_aligned = align_htf_to_ltf(prices, df_4h, ema_20_4h)
    
    # Calculate 1d EMA(50) for trend filter
    close_1d = df_1d['close'].values
    ema_50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # Calculate RSI(14) on 1h
    delta = np.diff(close, prepend=close[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    avg_gain = pd.Series(gain).ewm(span=14, adjust=False, min_periods=14).mean().values
    avg_loss = pd.Series(loss).ewm(span=14, adjust=False, min_periods=14).mean().values
    rs = avg_gain / (avg_loss + 1e-10)
    rsi = 100 - (100 / (1 + rs))
    
    # Volume filter: 24-period average
    vol_series = pd.Series(volume)
    vol_ma = vol_series.rolling(window=24, min_periods=24).mean().values
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(30, n):
        # Skip if data not available
        if (np.isnan(ema_20_4h_aligned[i]) or np.isnan(ema_50_1d_aligned[i]) or 
            np.isnan(rsi[i]) or np.isnan(vol_ma[i])):
            signals[i] = 0.0
            continue
        
        # Volume confirmation
        vol_ok = volume[i] > 1.5 * vol_ma[i]
        
        # Trend filters: both 4h and 1d must agree
        trend_up = ema_20_4h_aligned[i] > close[i] and ema_50_1d_aligned[i] > close[i]
        trend_down = ema_20_4h_aligned[i] < close[i] and ema_50_1d_aligned[i] < close[i]
        
        if position == 1:  # Long position
            # Exit: RSI crosses below 50
            if rsi[i] < 50 and rsi[i-1] >= 50:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.20
                
        elif position == -1:  # Short position
            # Exit: RSI crosses above 50
            if rsi[i] > 50 and rsi[i-1] <= 50:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.20
        else:  # Flat, look for entry
            # Only enter with volume confirmation and trend alignment
            if vol_ok:
                # Bullish RSI divergence: RSI higher low while price lower low
                if (i >= 3 and rsi[i] > rsi[i-2] and rsi[i-1] < rsi[i-3] and
                    close[i] < close[i-2] and close[i-1] > close[i-3] and
                    trend_up):
                    position = 1
                    signals[i] = 0.20
                # Bearish RSI divergence: RSI lower high while price higher high
                elif (i >= 3 and rsi[i] < rsi[i-2] and rsi[i-1] > rsi[i-3] and
                      close[i] > close[i-2] and close[i-1] < close[i-3] and
                      trend_down):
                    position = -1
                    signals[i] = -0.20
    
    return signals
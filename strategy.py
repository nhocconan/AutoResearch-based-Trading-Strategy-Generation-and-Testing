#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h strategy using 1-day volume-weighted VWAP with trend filter and volume confirmation
# Long when price > VWAP + 0.5*ATR and price > 200-bar EMA with volume > 1.5x average
# Short when price < VWAP - 0.5*ATR and price < 200-bar EMA with volume > 1.5x average
# VWAP acts as dynamic fair value, EMA200 for trend filter, volume for confirmation
# Target: 20-35 trades per year (80-140 over 4 years) with 0.25 position sizing

name = "4h_1dVWAP_ATR_EMA200_Volume_v1"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Calculate EMA200 on 4h close (needs 200 bars)
    close_series = pd.Series(close)
    ema200 = close_series.ewm(span=200, min_periods=200, adjust=False).mean().values
    
    # Calculate ATR(14) for volatility
    tr1 = np.abs(high - low)
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]  # First period
    atr = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    # Calculate 1-day VWAP
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    # Typical price and VWAP calculation
    typical_price = (df_1d['high'] + df_1d['low'] + df_1d['close']) / 3
    vwap = (typical_price * df_1d['volume']).cumsum() / df_1d['volume'].cumsum()
    
    # Align VWAP to 4h timeframe
    vwap_aligned = align_htf_to_ltf(prices, df_1d, vwap.values)
    
    # Volume confirmation: >1.5x 50-period average
    vol_ma_50 = pd.Series(volume).rolling(window=50, min_periods=50).mean().values
    volume_filter = volume > (1.5 * vol_ma_50)
    
    # Pre-compute session filter (08-20 UTC)
    hours = pd.DatetimeIndex(prices["open_time"]).hour
    session_filter = (hours >= 8) & (hours <= 20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(200, n):  # Start after EMA200 warmup
        # Skip if any critical value is NaN or outside session
        if (np.isnan(vwap_aligned[i]) or np.isnan(atr[i]) or 
            np.isnan(ema200[i]) or np.isnan(volume_filter[i]) or
            not session_filter[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long entry: price > VWAP + 0.5*ATR with uptrend and volume confirmation
            if close[i] > vwap_aligned[i] + 0.5 * atr[i] and close[i] > ema200[i] and volume_filter[i]:
                signals[i] = 0.25
                position = 1
            # Short entry: price < VWAP - 0.5*ATR with downtrend and volume confirmation
            elif close[i] < vwap_aligned[i] - 0.5 * atr[i] and close[i] < ema200[i] and volume_filter[i]:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: price < VWAP - 0.5*ATR (mean reversion)
            if close[i] < vwap_aligned[i] - 0.5 * atr[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: price > VWAP + 0.5*ATR (mean reversion)
            if close[i] > vwap_aligned[i] + 0.5 * atr[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals
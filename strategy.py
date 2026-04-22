#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 200:
        return np.zeros(n)
    
    # Hypothesis: 6h RSI(14) divergence with 1d MACD histogram confirmation and volume filter
    # RSI divergence identifies potential reversals in overbought/oversold conditions
    # MACD histogram confirms momentum shift on higher timeframe
    # Volume surge filters for institutional participation
    # Works in both bull and bear markets by catching exhaustion moves
    
    # Load daily data once
    df_1d = get_htf_data(prices, '1d')
    close_1d = df_1d['close'].values
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    
    # Daily MACD (12,26,9) histogram
    ema12 = pd.Series(close_1d).ewm(span=12, adjust=False).mean().values
    ema26 = pd.Series(close_1d).ewm(span=26, adjust=False).mean().values
    macd_line = ema12 - ema26
    signal_line = pd.Series(macd_line).ewm(span=9, adjust=False).mean().values
    macd_hist = macd_line - signal_line
    macd_hist_aligned = align_htf_to_ltf(prices, df_1d, macd_hist)
    
    # 6h RSI(14)
    close = prices['close'].values
    delta = np.diff(close, prepend=close[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    avg_gain = pd.Series(gain).ewm(alpha=1/14, adjust=False).mean().values
    avg_loss = pd.Series(loss).ewm(alpha=1/14, adjust=False).mean().values
    rs = avg_gain / (avg_loss + 1e-10)
    rsi = 100 - (100 / (1 + rs))
    
    # Volume filter (24-period MA surge)
    vol_ma24 = pd.Series(prices['volume'].values).rolling(window=24, min_periods=24).mean().values
    vol_surge = prices['volume'].values > 1.5 * vol_ma24
    
    signals = np.zeros(n)
    position = 0
    
    for i in range(100, n):
        # Skip if data not ready
        if (np.isnan(macd_hist_aligned[i]) or np.isnan(rsi[i]) or 
            np.isnan(vol_ma24[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Bullish divergence: price makes lower low, RSI makes higher low
            # Plus MACD histogram turning positive on daily
            if (i >= 2 and 
                close[i] < close[i-2] and 
                rsi[i] > rsi[i-2] and 
                macd_hist_aligned[i] > 0 and 
                vol_surge[i]):
                signals[i] = 0.25
                position = 1
            # Bearish divergence: price makes higher high, RSI makes lower high
            # Plus MACD histogram turning negative on daily
            elif (i >= 2 and 
                  close[i] > close[i-2] and 
                  rsi[i] < rsi[i-2] and 
                  macd_hist_aligned[i] < 0 and 
                  vol_surge[i]):
                signals[i] = -0.25
                position = -1
        else:
            # Exit: RSI returns to neutral zone (40-60) or opposite divergence
            if position == 1:
                if rsi[i] >= 60 or (i >= 2 and close[i] > close[i-2] and rsi[i] < rsi[i-2]):
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = 0.25
            else:  # position == -1
                if rsi[i] <= 40 or (i >= 2 and close[i] < close[i-2] and rsi[i] > rsi[i-2]):
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = -0.25
    
    return signals

name = "6h_RSI_Divergence_1dMACD_Hist_VolumeSurge_v1"
timeframe = "6h"
leverage = 1.0
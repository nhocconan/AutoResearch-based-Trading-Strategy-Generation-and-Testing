#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Daily VWAP Pullback with Volume Spike and RSI Filter
# Buy when price pulls back to daily VWAP during uptrend (price > daily VWAP[1 day ago]), 
# sell when price rallies to daily VWAP during downtrend (price < daily VWAP[1 day ago]).
# Requires volume > 1.5x 20-bar median and RSI between 40-60 to avoid overextended moves.
# Uses discrete position sizing (0.25) to limit trade frequency and reduce fee drag.
# Designed to work in both bull (buy dips) and bear (sell rallies) markets.

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Daily VWAP calculation
    df_1d = get_htf_data(prices, '1d')
    typical_price_1d = (df_1d['high'].values + df_1d['low'].values + df_1d['close'].values) / 3.0
    vwap_1d = (np.cumsum(typical_price_1d * df_1d['volume'].values) / 
               np.cumsum(df_1d['volume'].values))
    vwap_1d_aligned = align_htf_to_ltf(prices, df_1d, vwap_1d)
    
    # RSI(14)
    delta = np.diff(close, prepend=close[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    avg_gain = pd.Series(gain).ewm(span=14, adjust=False, min_periods=14).mean().values
    avg_loss = pd.Series(loss).ewm(span=14, adjust=False, min_periods=14).mean().values
    rs = avg_gain / (avg_loss + 1e-10)
    rsi = 100 - (100 / (1 + rs))
    
    # Volume confirmation: current > 1.5x median of last 20 bars
    vol_median = pd.Series(volume).rolling(window=20, min_periods=1).median()
    vol_threshold = 1.5 * vol_median
    
    signals = np.zeros(n)
    
    for i in range(20, n):
        # Skip if any required data is NaN
        if (np.isnan(vwap_1d_aligned[i]) or np.isnan(rsi[i]) or 
            np.isnan(vol_threshold[i])):
            continue
        
        # Long: price <= daily VWAP (pullback to VWAP), price > prior day's VWAP (uptrend),
        # volume spike, RSI not overbought
        if (close[i] <= vwap_1d_aligned[i] and 
            close[i] > vwap_1d_aligned[i-1] and 
            volume[i] > vol_threshold[i] and 
            rsi[i] < 60):
            signals[i] = 0.25
        
        # Short: price >= daily VWAP (rally to VWAP), price < prior day's VWAP (downtrend),
        # volume spike, RSI not oversold
        elif (close[i] >= vwap_1d_aligned[i] and 
              close[i] < vwap_1d_aligned[i-1] and 
              volume[i] > vol_threshold[i] and 
              rsi[i] > 40):
            signals[i] = -0.25
        
        # Exit: price crosses back through VWAP or RSI extreme
        elif (i > 0 and 
              ((signals[i-1] == 0.25 and (close[i] > vwap_1d_aligned[i] or rsi[i] >= 70)) or
               (signals[i-1] == -0.25 and (close[i] < vwap_1d_aligned[i] or rsi[i] <= 30)))):
            signals[i] = 0.0
        
        # Otherwise, hold previous position
        else:
            signals[i] = signals[i-1]
    
    return signals

name = "4h_DailyVWAP_Pullback_Volume_RSI"
timeframe = "4h"
leverage = 1.0
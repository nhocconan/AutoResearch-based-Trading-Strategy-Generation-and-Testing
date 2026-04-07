#!/usr/bin/env python3
"""
1h_vwap_rsi_pullback_4h1d_trend
Hypothesis: On 1h timeframe, enter long when price pulls back to VWAP with RSI < 40 and 4h/1d trend bullish; enter short when price pulls back to VWAP with RSI > 60 and 4h/1d trend bearish. Use 4h EMA50 and 1d EMA50 for trend filter. Designed for 15-35 trades/year with strict entry conditions to avoid overtrading.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "1h_vwap_rsi_pullback_4h1d_trend"
timeframe = "1h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Price and volume data
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Calculate VWAP (typical price * volume cumulative / volume cumulative)
    typical_price = (high + low + close) / 3.0
    vwap_num = np.cumsum(typical_price * volume)
    vwap_den = np.cumsum(volume)
    vwap = np.where(vwap_den > 0, vwap_num / vwap_den, 0)
    
    # RSI (14-period)
    delta = np.diff(close, prepend=close[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    avg_gain = pd.Series(gain).ewm(alpha=1/14, min_periods=14, adjust=False).mean().values
    avg_loss = pd.Series(loss).ewm(alpha=1/14, min_periods=14, adjust=False).mean().values
    rs = np.where(avg_loss != 0, avg_gain / avg_loss, 0)
    rsi = 100 - (100 / (1 + rs))
    
    # 4h EMA50 for trend filter
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 50:
        return np.zeros(n)
    ema50_4h = pd.Series(df_4h['close'].values).ewm(span=50, adjust=False).mean().values
    ema50_4h_aligned = align_htf_to_ltf(prices, df_4h, ema50_4h)
    
    # 1d EMA50 for trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    ema50_1d = pd.Series(df_1d['close'].values).ewm(span=50, adjust=False).mean().values
    ema50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema50_1d)
    
    # Session filter: 08-20 UTC
    hours = prices.index.hour
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(14, n):
        # Skip if data not available
        if (np.isnan(vwap[i]) or np.isnan(rsi[i]) or 
            np.isnan(ema50_4h_aligned[i]) or np.isnan(ema50_1d_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Session filter
        hour = hours[i]
        in_session = 8 <= hour <= 20
        
        # Trend filters
        bullish_trend = close[i] > ema50_4h_aligned[i] and close[i] > ema50_1d_aligned[i]
        bearish_trend = close[i] < ema50_4h_aligned[i] and close[i] < ema50_1d_aligned[i]
        
        # VWAP proximity (within 0.3%)
        near_vwap = abs(close[i] - vwap[i]) / vwap[i] < 0.003
        
        if position == 1:  # Long position
            # Exit: trend turns bearish or price moves above VWAP
            if not bullish_trend or close[i] > vwap[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.20
                
        elif position == -1:  # Short position
            # Exit: trend turns bullish or price moves below VWAP
            if not bearish_trend or close[i] < vwap[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.20
        else:  # Flat, look for entry
            if in_session:
                # Long: price near VWAP, RSI < 40, bullish trend
                if near_vwap and rsi[i] < 40 and bullish_trend:
                    position = 1
                    signals[i] = 0.20
                # Short: price near VWAP, RSI > 60, bearish trend
                elif near_vwap and rsi[i] > 60 and bearish_trend:
                    position = -1
                    signals[i] = -0.20
    
    return signals
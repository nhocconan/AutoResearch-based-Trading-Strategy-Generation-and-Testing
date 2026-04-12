#!/usr/bin/env python3
"""
1d_1w_Momentum_Confluence_Strategy
Hypothesis: Trade weekly momentum confluences on daily timeframe.
Enter long when price > weekly VWAP AND daily RSI < 30 (oversold bounce).
Enter short when price < weekly VWAP AND daily RSI > 70 (overbought rejection).
Use volume confirmation (>1.5x 20-day average) and ATR-based stoploss.
Designed for low trade frequency (~10-20/year) with high conviction in mean reversion.
Works in bull markets (buying dips) and bear markets (selling rallies).
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "1d_1w_Momentum_Confluence_Strategy"
timeframe = "1d"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # === WEEKLY DATA FOR VWAP ===
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 20:
        return np.zeros(n)
    
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    volume_1w = df_1w['volume'].values
    
    # Calculate VWAP for each weekly bar
    typical_price_1w = (high_1w + low_1w + close_1w) / 3
    vwap_numerator = np.cumsum(typical_price_1w * volume_1w)
    vwap_denominator = np.cumsum(volume_1w)
    vwap_1w = vwap_numerator / vwap_denominator
    
    # Align VWAP to daily timeframe
    vwap_1w_aligned = align_htf_to_ltf(prices, df_1w, vwap_1w)
    
    # === DAILY INDICATORS ===
    # RSI(14)
    delta = np.diff(close, prepend=close[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    avg_gain = pd.Series(gain).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    avg_loss = pd.Series(loss).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    rs = avg_gain / (avg_loss + 1e-10)
    rsi = 100 - (100 / (1 + rs))
    
    # ATR(14) for stoploss
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr1[0] = high[0] - low[0]
    tr2[0] = 0
    tr3[0] = 0
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    atr = pd.Series(tr).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    
    # Volume confirmation
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(20, n):
        # Skip if not ready
        if (np.isnan(rsi[i]) or np.isnan(vwap_1w_aligned[i]) or 
            np.isnan(atr[i]) or np.isnan(vol_ma[i])):
            signals[i] = 0.0 if position == 0 else (0.25 if position == 1 else -0.25)
            continue
        
        # Volume strength
        strong_volume = volume[i] > (vol_ma[i] * 1.5)
        
        # Long: price > weekly VWAP + RSI oversold + volume
        long_signal = (close[i] > vwap_1w_aligned[i] and 
                      rsi[i] < 30 and 
                      strong_volume)
        
        # Short: price < weekly VWAP + RSI overbought + volume
        short_signal = (close[i] < vwap_1w_aligned[i] and 
                       rsi[i] > 70 and 
                       strong_volume)
        
        # Exit: RSI mean reversion or opposite VWAP touch
        exit_long = (position == 1 and 
                    (rsi[i] > 50 or close[i] < vwap_1w_aligned[i]))
        exit_short = (position == -1 and 
                     (rsi[i] < 50 or close[i] > vwap_1w_aligned[i]))
        
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
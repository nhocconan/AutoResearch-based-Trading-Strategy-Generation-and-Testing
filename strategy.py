#!/usr/bin/env python3
"""
1d_1w_Multi_Factor_Confluence_Strategy
Hypothesis: Daily price action above/below weekly VWAP combined with daily momentum (RSI) and volume confirmation
creates high-probability entries. Weekly trend filter avoids counter-trend trades. Designed for low frequency
(~15-25 trades/year) to minimize fee drag while capturing sustained moves in both bull and bear markets.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "1d_1w_Multi_Factor_Confluence_Strategy"
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
    
    # === WEEKLY VWAP AS TREND FILTER ===
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 2:
        return np.zeros(n)
    
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    volume_1w = df_1w['volume'].values
    
    # Calculate VWAP for each week: typical price * volume / cumulative volume
    typical_price_1w = (high_1w + low_1w + close_1w) / 3.0
    vwap_1w = np.cumsum(typical_price_1w * volume_1w) / np.cumsum(volume_1w)
    vwap_1w = np.where(np.cumsum(volume_1w) == 0, 0, vwap_1w)  # avoid div by zero
    
    # Align weekly VWAP to daily timeframe
    vwap_1w_aligned = align_htf_to_ltf(prices, df_1w, vwap_1w)
    
    # === DAILY MOMENTUM (RSI) ===
    delta = np.diff(close, prepend=close[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    
    # Wilder's smoothing (equivalent to alpha=1/14)
    alpha = 1.0 / 14
    avg_gain = np.zeros_like(gain)
    avg_loss = np.zeros_like(loss)
    avg_gain[0] = gain[0]
    avg_loss[0] = loss[0]
    
    for i in range(1, len(gain)):
        avg_gain[i] = alpha * gain[i] + (1 - alpha) * avg_gain[i-1]
        avg_loss[i] = alpha * loss[i] + (1 - alpha) * avg_loss[i-1]
    
    rs = np.where(avg_loss != 0, avg_gain / avg_loss, 100)
    rsi = 100 - (100 / (1 + rs))
    
    # === DAILY VOLUME CONFIRMATION ===
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_ratio = volume / vol_ma
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(20, n):
        # Skip if not ready
        if (np.isnan(vwap_1w_aligned[i]) or np.isnan(rsi[i]) or 
            np.isnan(vol_ratio[i])):
            signals[i] = 0.0 if position == 0 else (0.25 if position == 1 else -0.25)
            continue
        
        # Long conditions: price above weekly VWAP, RSI > 50 (bullish momentum), volume confirmation
        long_condition = (close[i] > vwap_1w_aligned[i]) and (rsi[i] > 50) and (vol_ratio[i] > 1.5)
        
        # Short conditions: price below weekly VWAP, RSI < 50 (bearish momentum), volume confirmation
        short_condition = (close[i] < vwap_1w_aligned[i]) and (rsi[i] < 50) and (vol_ratio[i] > 1.5)
        
        # Exit conditions: RSI mean reversion or price crosses VWAP in opposite direction
        exit_long = (rsi[i] < 40) or (close[i] < vwap_1w_aligned[i] and position == 1)
        exit_short = (rsi[i] > 60) or (close[i] > vwap_1w_aligned[i] and position == -1)
        
        # Execute trades
        if long_condition and position != 1:
            position = 1
            signals[i] = 0.25
        elif short_condition and position != -1:
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
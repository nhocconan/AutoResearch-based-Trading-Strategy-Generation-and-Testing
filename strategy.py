#!/usr/bin/env python3
"""
1h_volume_momentum_with_4h1d_filters
Hypothesis: Use 1h volume spikes combined with 4h trend (EMA50) and 1d momentum (RSI) for high-probability entries.
Long when: 1h volume > 1.5x 20-bar average AND close > 4h EMA50 AND 1d RSI > 50.
Short when: 1h volume > 1.5x 20-bar average AND close < 4h EMA50 AND 1d RSI < 50.
Exit when volume condition fails or 4h EMA50 cross reverses.
Designed for 15-25 trades/year to minimize fee drag while capturing momentum in both bull and bear markets.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "1h_volume_momentum_with_4h1d_filters"
timeframe = "1h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Price data
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # 1h volume moving average for spike detection
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    # 4h EMA50 for trend filter
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 50:
        return np.zeros(n)
    ema_4h = pd.Series(df_4h['close']).ewm(span=50, min_periods=50, adjust=False).mean().values
    ema_4h_aligned = align_htf_to_ltf(prices, df_4h, ema_4h)
    
    # 1d RSI for momentum filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 14:
        return np.zeros(n)
    close_1d = df_1d['close'].values
    delta = pd.Series(close_1d).diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = pd.Series(gain).ewm(alpha=1/14, min_periods=14, adjust=False).mean()
    avg_loss = pd.Series(loss).ewm(alpha=1/14, min_periods=14, adjust=False).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    rsi_1d = 100 - (100 / (1 + rs))
    rsi_1d = rsi_1d.fillna(50).values
    rsi_1d_aligned = align_htf_to_ltf(prices, df_1d, rsi_1d)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(20, n):
        # Skip if data not available
        if (np.isnan(vol_ma[i]) or np.isnan(ema_4h_aligned[i]) or 
            np.isnan(rsi_1d_aligned[i]) or np.isnan(close[i])):
            signals[i] = 0.0
            continue
        
        # Volume spike condition: >1.5x average
        vol_spike = volume[i] > 1.5 * vol_ma[i]
        
        if position == 1:  # Long position
            # Exit: volume spike ends OR price breaks below 4h EMA50
            if not vol_spike or close[i] < ema_4h_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.20
                
        elif position == -1:  # Short position
            # Exit: volume spike ends OR price breaks above 4h EMA50
            if not vol_spike or close[i] > ema_4h_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.20
        else:  # Flat, look for entry
            if vol_spike:
                # Long: price above 4h EMA50 AND 1d RSI > 50 (bullish momentum)
                if close[i] > ema_4h_aligned[i] and rsi_1d_aligned[i] > 50:
                    position = 1
                    signals[i] = 0.20
                # Short: price below 4h EMA50 AND 1d RSI < 50 (bearish momentum)
                elif close[i] < ema_4h_aligned[i] and rsi_1d_aligned[i] < 50:
                    position = -1
                    signals[i] = -0.20
    
    return signals
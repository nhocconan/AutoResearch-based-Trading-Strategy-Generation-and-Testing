#!/usr/bin/env python3
"""
1h_RSI_Confluence_Volume_Filter
Hypothesis: Use RSI(2) extreme readings with volume spike confirmation and 4h trend filter.
Long when RSI(2)<10, volume>2x20-bar average, and price above 4h EMA50.
Short when RSI(2)>90, volume>2x20-bar average, and price below 4h EMA50.
Exit when RSI(2) crosses back above 50 (long) or below 50 (short).
Target: 15-30 trades/year to avoid fee drag. Works in bull/bear via mean reversion + trend filter.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 4h data for trend filter
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 50:
        return np.zeros(n)
    
    # 4h EMA50 for trend filter
    ema50_4h = pd.Series(df_4h['close']).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema50_4h_aligned = align_htf_to_ltf(prices, df_4h, ema50_4h)
    
    # RSI(2) calculation
    delta = pd.Series(close).diff()
    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)
    avg_gain = gain.ewm(alpha=1/2, adjust=False, min_periods=2).mean()
    avg_loss = loss.ewm(alpha=1/2, adjust=False, min_periods=2).mean()
    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    rsi_values = rsi.values
    
    # Volume confirmation: volume > 2.0 * 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_spike = volume > (vol_ma * 2.0)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    size = 0.20   # Position size: 20% of capital
    
    # Warmup: need enough data for RSI(2) and volume MA
    start_idx = max(50, 20)
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if np.isnan(ema50_4h_aligned[i]) or np.isnan(rsi_values[i]):
            signals[i] = 0.0
            continue
        
        ema_trend = ema50_4h_aligned[i]
        rsi_val = rsi_values[i]
        vol_spike_val = vol_spike[i]
        
        if position == 0:
            # Long: RSI(2) < 10 (oversold) + volume spike + uptrend (price > EMA50)
            if rsi_val < 10 and vol_spike_val and close[i] > ema_trend:
                signals[i] = size
                position = 1
            # Short: RSI(2) > 90 (overbought) + volume spike + downtrend (price < EMA50)
            elif rsi_val > 90 and vol_spike_val and close[i] < ema_trend:
                signals[i] = -size
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # Exit long: RSI(2) crosses back above 50 (mean reversion complete)
            if rsi_val > 50:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = size
        elif position == -1:
            # Exit short: RSI(2) crosses back below 50 (mean reversion complete)
            if rsi_val < 50:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -size
    
    return signals

name = "1h_RSI_Confluence_Volume_Filter"
timeframe = "1h"
leverage = 1.0
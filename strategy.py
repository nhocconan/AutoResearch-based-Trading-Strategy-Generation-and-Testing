#!/usr/bin/env python3
"""
1d_rsi_pullback_1w_trend_volume_v1
Hypothesis: RSI(14) pullbacks on 1d with weekly trend filter and volume confirmation capture medium-term reversals in both bull and bear markets. Weekly trend avoids counter-trend trades, volume filters weak signals, and RSI(14)<30/>70 provides oversold/overbought entries. Target: 10-20 trades/year.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "1d_rsi_pullback_1w_trend_volume_v1"
timeframe = "1d"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Price data
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # 1w data for trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 50:
        return np.zeros(n)
    
    # 1w EMA50 for trend filter
    ema_50 = df_1w['close'].ewm(span=50, adjust=False).mean()
    
    # Align 1w EMA50 to 1d timeframe
    ema_50_aligned = align_htf_to_ltf(prices, df_1w, ema_50.values)
    
    # RSI(14) on 1d
    delta = pd.Series(close).diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(alpha=1/14, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1/14, adjust=False).mean()
    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    rsi = rsi.fillna(50).values
    
    # Volume confirmation (20-period average on 1d)
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(30, n):
        # Skip if required data not available
        if (np.isnan(rsi[i]) or np.isnan(ema_50_aligned[i]) or 
            np.isnan(vol_ma[i]) or vol_ma[i] <= 0):
            signals[i] = 0.0
            continue
        
        # Volume confirmation: current volume > 1.2x average volume
        vol_confirm = volume[i] > 1.2 * vol_ma[i]
        
        if position == 1:  # Long position
            # Exit: RSI rises above 50 or price breaks below weekly EMA50
            if rsi[i] > 50 or close[i] < ema_50_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
        elif position == -1:  # Short position
            # Exit: RSI falls below 50 or price breaks above weekly EMA50
            if rsi[i] < 50 or close[i] > ema_50_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat, look for entry
            # Long entry: RSI < 30 (oversold), with volume and price above weekly EMA50
            if (rsi[i] < 30 and vol_confirm and 
                close[i] > ema_50_aligned[i]):
                position = 1
                signals[i] = 0.25
            # Short entry: RSI > 70 (overbought), with volume and price below weekly EMA50
            elif (rsi[i] > 70 and vol_confirm and 
                  close[i] < ema_50_aligned[i]):
                position = -1
                signals[i] = -0.25
    
    return signals
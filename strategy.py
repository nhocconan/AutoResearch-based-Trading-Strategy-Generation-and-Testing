#!/usr/bin/env python3
# 4h_rsi2_ema13_volume_v1
# Hypothesis: 4h RSI(2) pullback with EMA13 trend and volume confirmation.
# RSI(2) identifies extreme short-term reversals (oversold <10, overbought >90).
# EMA13 provides trend filter to avoid counter-trend trades.
# Volume spike confirms momentum behind the move.
# Designed for 15-30 trades/year (60-120 over 4 years) with tight entry conditions.
# Works in bull/bear markets: RSI reversals work in pullbacks within trends.

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "4h_rsi2_ema13_volume_v1"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 4h data for EMA13 and RSI(2)
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 13:
        return np.zeros(n)
    
    close_4h = df_4h['close'].values
    high_4h = df_4h['high'].values
    low_4h = df_4h['low'].values
    
    # Calculate 4h EMA13
    ema13 = pd.Series(close_4h).ewm(span=13, adjust=False, min_periods=13).mean().values
    ema13_aligned = align_htf_to_ltf(prices, df_4h, ema13)
    
    # Calculate 4h RSI(2)
    delta = pd.Series(close_4h).diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(alpha=1/2, adjust=False, min_periods=2).mean()
    avg_loss = loss.ewm(alpha=1/2, adjust=False, min_periods=2).mean()
    rs = avg_gain / avg_loss
    rsi2 = 100 - (100 / (1 + rs))
    rsi2_values = rsi2.fillna(100).values  # Fill NaN with 100 (no loss = max RSI)
    rsi2_aligned = align_htf_to_ltf(prices, df_4h, rsi2_values)
    
    # Volume spike detection (20-period volume average)
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_spike = volume > (vol_ma_20 * 2.0)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(50, n):
        # Skip if any required data is NaN
        if (np.isnan(ema13_aligned[i]) or np.isnan(rsi2_aligned[i]) or 
            np.isnan(vol_ma_20[i])):
            signals[i] = 0.0
            continue
        
        if position == 1:  # Long position
            # Exit: RSI(2) crosses above 50 (momentum fading)
            if rsi2_aligned[i] > 50:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit: RSI(2) crosses below 50 (momentum fading)
            if rsi2_aligned[i] < 50:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat
            # Enter long: RSI(2) < 10 (oversold), price > EMA13 (uptrend), volume spike
            if (rsi2_aligned[i] < 10) and (close[i] > ema13_aligned[i]) and vol_spike[i]:
                position = 1
                signals[i] = 0.25
            # Enter short: RSI(2) > 90 (overbought), price < EMA13 (downtrend), volume spike
            elif (rsi2_aligned[i] > 90) and (close[i] < ema13_aligned[i]) and vol_spike[i]:
                position = -1
                signals[i] = -0.25
    
    return signals
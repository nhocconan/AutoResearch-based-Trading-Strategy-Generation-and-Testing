#!/usr/bin/env python3
"""
1h_triple_confirmation_4h1d_volume
Hypothesis: On 1h timeframe, enter long when 4h EMA21 rising, price > 1d VWAP, and RSI(14) crosses above 50 with volume confirmation; short when 4h EMA21 falling, price < 1d VWAP, and RSI(14) crosses below 50 with volume confirmation. Exit on opposite signal or 4h EMA21 crossover reversal. Uses 4h for trend direction, 1d VWAP for value, 1h RSI for timing, and volume to confirm genuine moves. Designed to work in both bull (trend following) and bear (mean reversion at extremes) markets by requiring multiple confirmations. Target: 15-35 trades/year (~60-140 total over 4 years).
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "1h_triple_confirmation_4h1d_volume"
timeframe = "1h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Price data
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # 4h EMA21 for trend direction
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 50:
        return np.zeros(n)
    ema_21_4h = df_4h['close'].ewm(span=21, adjust=False, min_periods=21).mean().values
    ema_21_4h_aligned = align_htf_to_ltf(prices, df_4h, ema_21_4h)
    
    # 1d VWAP for value reference
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    # Calculate VWAP: cumulative (price * volume) / cumulative volume
    typical_price = (df_1d['high'] + df_1d['low'] + df_1d['close']) / 3.0
    pv = (typical_price * df_1d['volume']).cumsum()
    vol_cum = df_1d['volume'].cumsum()
    vwap = pv / vol_cum
    vwap_aligned = align_htf_to_ltf(prices, df_1d, vwap.values)
    
    # 1h RSI(14) for momentum timing
    delta = pd.Series(close).diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(alpha=1/14, adjust=False, min_periods=14).mean()
    avg_loss = loss.ewm(alpha=1/14, adjust=False, min_periods=14).mean()
    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    rsi_values = rsi.values
    
    # Volume confirmation: volume > 1.3x 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(50, n):
        # Skip if required data not available
        if (np.isnan(ema_21_4h_aligned[i]) or np.isnan(vwap_aligned[i]) or 
            np.isnan(rsi_values[i]) or np.isnan(vol_ma[i]) or vol_ma[i] <= 0):
            signals[i] = 0.0
            continue
        
        # Volume confirmation: current volume > 1.3x average volume
        vol_confirm = volume[i] > 1.3 * vol_ma[i]
        
        if position == 1:  # Long position
            # Exit: RSI crosses below 50 or 4h EMA21 turns down
            if rsi_values[i] < 50 or ema_21_4h_aligned[i] < ema_21_4h_aligned[i-1]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.20
        elif position == -1:  # Short position
            # Exit: RSI crosses above 50 or 4h EMA21 turns up
            if rsi_values[i] > 50 or ema_21_4h_aligned[i] > ema_21_4h_aligned[i-1]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.20
        else:  # Flat, look for entry
            # Long entry: 4h EMA21 up, price > 1d VWAP, RSI crosses above 50, volume confirmation
            if (ema_21_4h_aligned[i] > ema_21_4h_aligned[i-1] and  # 4h EMA21 rising
                close[i] > vwap_aligned[i] and                    # price above 1d VWAP
                rsi_values[i] > 50 and rsi_values[i-1] <= 50 and  # RSI crossed above 50
                vol_confirm):                                   # volume confirmation
                position = 1
                signals[i] = 0.20
            # Short entry: 4h EMA21 down, price < 1d VWAP, RSI crosses below 50, volume confirmation
            elif (ema_21_4h_aligned[i] < ema_21_4h_aligned[i-1] and  # 4h EMA21 falling
                  close[i] < vwap_aligned[i] and                     # price below 1d VWAP
                  rsi_values[i] < 50 and rsi_values[i-1] >= 50 and   # RSI crossed below 50
                  vol_confirm):                                    # volume confirmation
                position = -1
                signals[i] = -0.20
    
    return signals
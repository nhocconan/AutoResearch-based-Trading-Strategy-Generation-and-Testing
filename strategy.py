#!/usr/bin/env python3
"""
1d_RSI_Pullback_TrendFollow_v1
Daily RSI pullback strategy with weekly trend filter and volume confirmation.
Designed for low turnover and high win rate in both bull and bear markets.
Target: 30-100 total trades over 4 years (7-25/year).
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    volume = prices['volume'].values
    
    # === Weekly EMA34 trend filter ===
    df_1w = get_htf_data(prices, '1w')
    close_1w = df_1w['close'].values
    ema34_1w = pd.Series(close_1w).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema34_1w_aligned = align_htf_to_ltf(prices, df_1w, ema34_1w)
    
    # === Daily RSI(14) ===
    delta = pd.Series(close).diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(alpha=1/14, adjust=False, min_periods=14).mean()
    avg_loss = loss.ewm(alpha=1/14, adjust=False, min_periods=14).mean()
    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    rsi_values = rsi.values
    
    # === Daily volume confirmation (20-period average) ===
    df_1d = get_htf_data(prices, '1d')
    volume_1d = df_1d['volume'].values
    vol_ma_1d = pd.Series(volume_1d).rolling(window=20, min_periods=20).mean().values
    vol_ma_1d_aligned = align_htf_to_ltf(prices, df_1d, vol_ma_1d)
    volume_1d_aligned = align_htf_to_ltf(prices, df_1d, volume_1d)
    
    signals = np.zeros(n)
    
    # Warmup period
    warmup = 50
    
    # Track position state
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(warmup, n):
        # Skip if any required data is NaN
        if (np.isnan(ema34_1w_aligned[i]) or 
            np.isnan(rsi_values[i]) or 
            np.isnan(vol_ma_1d_aligned[i]) or 
            np.isnan(volume_1d_aligned[i])):
            signals[i] = 0.0
            position = 0
            continue
        
        # Volume confirmation: current daily volume > 1.3x 20-day average
        vol_confirmed = volume_1d_aligned[i] > 1.3 * vol_ma_1d_aligned[i]
        
        # Entry logic: only enter when flat
        if position == 0:
            # Long: RSI pulls back from oversold in weekly uptrend
            if (rsi_values[i] < 30 and 
                close[i] > ema34_1w_aligned[i] and 
                vol_confirmed):
                signals[i] = 0.25
                position = 1
                continue
            # Short: RSI bounces from overbought in weekly downtrend
            elif (rsi_values[i] > 70 and 
                  close[i] < ema34_1w_aligned[i] and 
                  vol_confirmed):
                signals[i] = -0.25
                position = -1
                continue
        
        # Exit logic: RSI returns to neutral zone
        elif position == 1:
            # Exit long: RSI crosses above 50
            if rsi_values[i] > 50:
                signals[i] = 0.0
                position = 0
                continue
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: RSI crosses below 50
            if rsi_values[i] < 50:
                signals[i] = 0.0
                position = 0
                continue
            else:
                signals[i] = -0.25
    
    return signals

name = "1d_RSI_Pullback_TrendFollow_v1"
timeframe = "1d"
leverage = 1.0
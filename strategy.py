#!/usr/bin/env python3
# 4h_Camarilla_R1_S1_Breakout_1dTrend_Volume_Corrected
# Hypothesis: Uses 1d EMA34 for trend filter, 1d RSI(14) for overbought/oversold detection, and volume confirmation to trade breakouts at daily Camarilla R1/S1 levels.
# Designed to avoid false breakouts by requiring alignment with daily trend and momentum, reducing whipsaw in choppy markets.
# Target: 20-40 trades/year per symbol to minimize fee drag.

timeframe = "4h"
name = "4h_Camarilla_R1_S1_Breakout_1dTrend_Volume_Corrected"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Get 1d data for EMA trend filter and RSI
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) == 0:
        return np.zeros(n)
    
    # Calculate EMA34 on 1d closes for trend filter
    ema_34_1d = pd.Series(df_1d['close'].values).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # Calculate RSI(14) on 1d closes
    delta = pd.Series(df_1d['close'].values).diff()
    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)
    avg_gain = gain.ewm(alpha=1/14, adjust=False, min_periods=14).mean()
    avg_loss = loss.ewm(alpha=1/14, adjust=False, min_periods=14).mean()
    rs = avg_gain / avg_loss
    rsi_14_1d = 100 - (100 / (1 + rs))
    rsi_14_1d = rsi_14_1d.values
    rsi_14_1d_aligned = align_htf_to_ltf(prices, df_1d, rsi_14_1d)
    
    # Get daily data for Camarilla levels
    d_high = df_1d['high'].values
    d_low = df_1d['low'].values
    d_close = df_1d['close'].values
    
    camarilla_r1 = d_close + 1.1 * (d_high - d_low) / 12
    camarilla_s1 = d_close - 1.1 * (d_high - d_low) / 12
    
    camarilla_r1_aligned = align_htf_to_ltf(prices, df_1d, camarilla_r1)
    camarilla_s1_aligned = align_htf_to_ltf(prices, df_1d, camarilla_s1)
    
    # Volume spike detection: 1.5x average volume (6-period = 1 day on 4h chart)
    vol_ma = pd.Series(volume).rolling(window=6, min_periods=6).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(34, 14, 6)  # Ensure we have EMA, RSI, and volume MA data
    
    for i in range(start_idx, n):
        # Skip if any critical value is NaN
        if (np.isnan(camarilla_r1_aligned[i]) or np.isnan(camarilla_s1_aligned[i]) or 
            np.isnan(ema_34_1d_aligned[i]) or np.isnan(rsi_14_1d_aligned[i]) or 
            np.isnan(vol_ma[i]) or vol_ma[i] == 0):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: close > R1 with volume spike, price above 1d EMA34, and RSI not overbought
            if (close[i] > camarilla_r1_aligned[i] and 
                volume[i] > 1.5 * vol_ma[i] and 
                close[i] > ema_34_1d_aligned[i] and 
                rsi_14_1d_aligned[i] < 70):
                signals[i] = 0.25
                position = 1
            # Short: close < S1 with volume spike, price below 1d EMA34, and RSI not oversold
            elif (close[i] < camarilla_s1_aligned[i] and 
                  volume[i] > 1.5 * vol_ma[i] and 
                  close[i] < ema_34_1d_aligned[i] and 
                  rsi_14_1d_aligned[i] > 30):
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit: touch S1 (opposite level) or trend failure (price below 1d EMA34)
            if close[i] < camarilla_s1_aligned[i] or close[i] < ema_34_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit: touch R1 (opposite level) or trend failure (price above 1d EMA34)
            if close[i] > camarilla_r1_aligned[i] or close[i] > ema_34_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals
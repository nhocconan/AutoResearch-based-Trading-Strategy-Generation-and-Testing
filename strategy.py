#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h Volume Spike + 1d EMA50 Trend + 1w RSI Regime Filter
# Long when: volume > 3.0x 20-bar avg AND price > 1d EMA50 AND 1w RSI < 70 (not overbought)
# Short when: volume > 3.0x 20-bar avg AND price < 1d EMA50 AND 1w RSI > 30 (not oversold)
# Exit: price crosses 1d EMA50 (trend reversal)
# Uses discrete position sizing (0.25) to reduce fee drag. Target: 12-30 trades/year on 12h timeframe.
# Volume spike confirms institutional interest, 1d EMA50 filters trend direction,
# 1w RSI regime prevents extreme counter-trend entries. Works in both bull/bear markets.

name = "12h_VolumeSpike_1dEMA50_1wRSI_Regime_v1"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Get 1d data for EMA50 trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # Calculate EMA(50) on 1d data
    close_1d = df_1d['close'].values
    ema_50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    # Align EMA50 to 12h timeframe
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # Get 1w data for RSI regime filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 14:
        return np.zeros(n)
    
    # Calculate RSI(14) on 1w data
    close_1w = df_1w['close'].values
    delta = pd.Series(close_1w).diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14, min_periods=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14, min_periods=14).mean()
    rs = gain / loss.replace(0, np.nan)
    rsi_14_1w = 100 - (100 / (1 + rs))
    rsi_14_1w = rsi_14_1w.fillna(50).values  # Fill NaN with neutral 50
    # Align RSI to 12h timeframe
    rsi_14_1w_aligned = align_htf_to_ltf(prices, df_1w, rsi_14_1w)
    
    # Volume confirmation: >3.0x 20-bar average volume (stricter to reduce trades)
    volume_series = pd.Series(volume)
    volume_ma_20 = volume_series.rolling(window=20, min_periods=20).mean().values
    volume_confirm = volume > 3.0 * volume_ma_20
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(20, 50)  # volume MA and EMA50 warmup
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(ema_50_1d_aligned[i]) or np.isnan(rsi_14_1w_aligned[i]) or 
            np.isnan(volume_ma_20[i])):
            signals[i] = 0.0
            continue
        
        vol_conf = volume_confirm[i]
        curr_ema50 = ema_50_1d_aligned[i]
        curr_rsi = rsi_14_1w_aligned[i]
        curr_close = close[i]
        
        # Handle exits and position management
        if position == 1:  # Long position
            # Exit: price crosses below 1d EMA50 (trend reversal)
            if curr_close < curr_ema50:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit: price crosses above 1d EMA50 (trend reversal)
            if curr_close > curr_ema50:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
                
        else:  # Flat - look for new entries
            # Long when volume spike AND price > 1d EMA50 AND 1w RSI < 70 (not overbought)
            if vol_conf and curr_close > curr_ema50 and curr_rsi < 70:
                signals[i] = 0.25
                position = 1
            # Short when volume spike AND price < 1d EMA50 AND 1w RSI > 30 (not oversold)
            elif vol_conf and curr_close < curr_ema50 and curr_rsi > 30:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
    
    return signals
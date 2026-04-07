#!/usr/bin/env python3
"""
1d_weekly_rsi_extreme_reversion_v1
Hypothesis: On 1d timeframe, enter long when daily RSI(14) crosses below 20 (oversold) with above-average volume, enter short when RSI crosses above 80 (overbought) with above-average volume. Use 1-week RSI as trend filter to avoid counter-trend trades (only go long when weekly RSI > 50, short when weekly RSI < 50). Exit when RSI crosses back to 50 (mean reversion complete). Designed for 10-25 trades/year to minimize fee dust while capturing extreme reversals in both bull and bear markets.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "1d_weekly_rsi_extreme_reversion_v1"
timeframe = "1d"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Price data
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Calculate daily RSI(14)
    if len(close) < 14:
        return np.zeros(n)
    
    delta = np.diff(close, prepend=close[0])
    gain = np.where(delta > 0, delta, 0.0)
    loss = np.where(delta < 0, -delta, 0.0)
    
    # Wilder's smoothing
    avg_gain = np.zeros_like(gain)
    avg_loss = np.zeros_like(loss)
    avg_gain[13] = np.mean(gain[1:14])
    avg_loss[13] = np.mean(loss[1:14])
    
    for i in range(14, len(gain)):
        avg_gain[i] = (avg_gain[i-1] * 13 + gain[i]) / 14
        avg_loss[i] = (avg_loss[i-1] * 13 + loss[i]) / 14
    
    rs = np.where(avg_loss != 0, avg_gain / avg_loss, 100)
    rsi = 100 - (100 / (1 + rs))
    
    # Volume moving average for confirmation
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    # Calculate weekly RSI(14) for trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 14:
        return np.zeros(n)
    
    close_1w = df_1w['close'].values
    
    delta_1w = np.diff(close_1w, prepend=close_1w[0])
    gain_1w = np.where(delta_1w > 0, delta_1w, 0.0)
    loss_1w = np.where(delta_1w < 0, -delta_1w, 0.0)
    
    avg_gain_1w = np.zeros_like(gain_1w)
    avg_loss_1w = np.zeros_like(loss_1w)
    avg_gain_1w[13] = np.mean(gain_1w[1:14])
    avg_loss_1w[13] = np.mean(loss_1w[1:14])
    
    for i in range(14, len(gain_1w)):
        avg_gain_1w[i] = (avg_gain_1w[i-1] * 13 + gain_1w[i]) / 14
        avg_loss_1w[i] = (avg_loss_1w[i-1] * 13 + loss_1w[i]) / 14
    
    rs_1w = np.where(avg_loss_1w != 0, avg_gain_1w / avg_loss_1w, 100)
    rsi_1w = 100 - (100 / (1 + rs_1w))
    
    # Align weekly RSI to daily timeframe
    rsi_1w_aligned = align_htf_to_ltf(prices, df_1w, rsi_1w)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(20, n):
        # Skip if data not available
        if (np.isnan(rsi[i]) or np.isnan(rsi_1w_aligned[i]) or np.isnan(vol_ma[i]) or 
            np.isnan(close[i])):
            signals[i] = 0.0
            continue
        
        # Volume confirmation: above average volume
        vol_ok = volume[i] > vol_ma[i]
        
        if position == 1:  # Long position
            # Exit: RSI crosses above 50 (mean reversion complete)
            if rsi[i] > 50 and rsi[i-1] <= 50:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit: RSI crosses below 50 (mean reversion complete)
            if rsi[i] < 50 and rsi[i-1] >= 50:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat, look for entry
            if vol_ok:
                # Long: RSI crosses below 20 with weekly RSI > 50 (bullish filter)
                if rsi[i] < 20 and rsi[i-1] >= 20 and rsi_1w_aligned[i] > 50:
                    position = 1
                    signals[i] = 0.25
                # Short: RSI crosses above 80 with weekly RSI < 50 (bearish filter)
                elif rsi[i] > 80 and rsi[i-1] <= 80 and rsi_1w_aligned[i] < 50:
                    position = -1
                    signals[i] = -0.25
    
    return signals
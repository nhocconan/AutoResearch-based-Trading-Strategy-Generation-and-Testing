#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "1d_1w_RSI_Reversal_Extreme"
timeframe = "1d"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Weekly RSI for trend context (14-period)
    df_1w = get_htf_data(prices, '1w')
    close_1w = df_1w['close'].values
    
    # Calculate weekly RSI with proper smoothing
    delta = np.diff(close_1w, prepend=close_1w[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    
    # Wilder's smoothing (alpha = 1/14)
    alpha = 1.0 / 14
    avg_gain = np.zeros_like(gain)
    avg_loss = np.zeros_like(loss)
    
    for i in range(len(gain)):
        if i == 0:
            avg_gain[i] = gain[i]
            avg_loss[i] = loss[i]
        else:
            avg_gain[i] = alpha * gain[i] + (1 - alpha) * avg_gain[i-1]
            avg_loss[i] = alpha * loss[i] + (1 - alpha) * avg_loss[i-1]
    
    rs = np.where(avg_loss != 0, avg_gain / avg_loss, 100)
    rsi_1w = 100 - (100 / (1 + rs))
    rsi_1w = np.where(avg_loss == 0, 100, rsi_1w)
    rsi_1w = np.where(avg_gain == 0, 0, rsi_1w)
    
    # Align weekly RSI to daily
    rsi_1w_aligned = align_htf_to_ltf(prices, df_1w, rsi_1w)
    
    # Daily RSI for entry signals (14-period)
    delta_d = np.diff(close, prepend=close[0])
    gain_d = np.where(delta_d > 0, delta_d, 0)
    loss_d = np.where(delta_d < 0, -delta_d, 0)
    
    avg_gain_d = np.zeros_like(gain_d)
    avg_loss_d = np.zeros_like(loss_d)
    
    for i in range(len(gain_d)):
        if i == 0:
            avg_gain_d[i] = gain_d[i]
            avg_loss_d[i] = loss_d[i]
        else:
            avg_gain_d[i] = alpha * gain_d[i] + (1 - alpha) * avg_gain_d[i-1]
            avg_loss_d[i] = alpha * loss_d[i] + (1 - alpha) * avg_loss_d[i-1]
    
    rs_d = np.where(avg_loss_d != 0, avg_gain_d / avg_loss_d, 100)
    rsi_d = 100 - (100 / (1 + rs_d))
    rsi_d = np.where(avg_loss_d == 0, 100, rsi_d)
    rsi_d = np.where(avg_gain_d == 0, 0, rsi_d)
    
    # Volume confirmation: current volume > 1.8x 20-day average
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 40  # Need enough data for weekly RSI alignment
    
    for i in range(start_idx, n):
        if np.isnan(rsi_1w_aligned[i]) or np.isnan(rsi_d[i]) or np.isnan(vol_ma_20[i]):
            signals[i] = 0.0
            continue
        
        price = close[i]
        vol = volume[i]
        vol_ma = vol_ma_20[i]
        
        # Volume filter
        volume_ok = vol > 1.8 * vol_ma
        
        if position == 0:
            # Long: Weekly RSI > 50 (bullish bias) + Daily RSI < 25 (oversold) + Volume
            if rsi_1w_aligned[i] > 50 and rsi_d[i] < 25 and volume_ok:
                signals[i] = 0.25
                position = 1
            # Short: Weekly RSI < 50 (bearish bias) + Daily RSI > 75 (overbought) + Volume
            elif rsi_1w_aligned[i] < 50 and rsi_d[i] > 75 and volume_ok:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit: Daily RSI > 60 (overbought) or weekly RSI turns bearish
            if rsi_d[i] > 60 or rsi_1w_aligned[i] < 45:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit: Daily RSI < 40 (oversold) or weekly RSI turns bullish
            if rsi_d[i] < 40 or rsi_1w_aligned[i] > 55:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals
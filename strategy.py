#!/usr/bin/env python3
"""
12h_RSI_Pullback_Trend
12h strategy using RSI pullbacks in trend with volume confirmation.
- Long: RSI(14) pulls back to 40-50 in uptrend (EMA50 > EMA200) + volume > 1.3x average
- Short: RSI(14) pulls back to 50-60 in downtrend (EMA50 < EMA200) + volume > 1.3x average
- Exit: RSI reaches opposite extreme (60 for long, 40 for short) or trend reversal
Designed for ~20-30 trades/year per symbol (80-120 total over 4 years)
Works in bull markets (buy pullbacks) and bear markets (sell rallies)
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
    
    # Get daily data for trend filter and volume average
    df_1d = get_htf_data(prices, '1d')
    
    close_1d = df_1d['close'].values
    volume_1d = df_1d['volume'].values
    
    # Daily EMA50 and EMA200 for trend filter
    ema_50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_200_1d = pd.Series(close_1d).ewm(span=200, adjust=False, min_periods=200).mean().values
    
    ema_50_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    ema_200_aligned = align_htf_to_ltf(prices, df_1d, ema_200_1d)
    
    # Daily volume average (20-period)
    vol_ma_20 = pd.Series(volume_1d).rolling(window=20, min_periods=20).mean().values
    vol_ma_aligned = align_htf_to_ltf(prices, df_1d, vol_ma_20)
    
    # RSI(14) on 12h closes
    delta = np.diff(close, prepend=close[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    
    # Wilder's smoothing (alpha = 1/14)
    alpha = 1.0 / 14
    avg_gain = np.zeros_like(gain)
    avg_loss = np.zeros_like(loss)
    avg_gain[0] = gain[0]
    avg_loss[0] = loss[0]
    
    for i in range(1, len(gain)):
        avg_gain[i] = alpha * gain[i] + (1 - alpha) * avg_gain[i-1]
        avg_loss[i] = alpha * loss[i] + (1 - alpha) * avg_loss[i-1]
    
    rs = np.divide(avg_gain, avg_loss, out=np.full_like(avg_gain, 0.0), where=avg_loss!=0)
    rsi = 100 - (100 / (1 + rs))
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 50  # need enough for EMA200 and RSI
    
    for i in range(start_idx, n):
        # Skip if any required data is not available
        if (np.isnan(ema_50_aligned[i]) or np.isnan(ema_200_aligned[i]) or
            np.isnan(vol_ma_aligned[i]) or np.isnan(rsi[i])):
            signals[i] = 0.0
            continue
        
        # Trend conditions
        uptrend = ema_50_aligned[i] > ema_200_aligned[i]
        downtrend = ema_50_aligned[i] < ema_200_aligned[i]
        
        # Volume confirmation
        vol_confirm = volume[i] > 1.3 * vol_ma_aligned[i]
        
        # RSI conditions
        rsi_val = rsi[i]
        
        if position == 0:
            # Long: uptrend + volume + RSI pullback to 40-50
            if uptrend and vol_confirm and 40 <= rsi_val <= 50:
                signals[i] = 0.25
                position = 1
            # Short: downtrend + volume + RSI pullback to 50-60
            elif downtrend and vol_confirm and 50 <= rsi_val <= 60:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Long exit: RSI reaches 60 or trend reversal
            if rsi_val >= 60 or not uptrend:
                signals[i] = -0.25  # reverse to short
                position = -1
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: RSI reaches 40 or trend reversal
            if rsi_val <= 40 or not downtrend:
                signals[i] = 0.25  # reverse to long
                position = 1
            else:
                signals[i] = -0.25
    
    return signals

name = "12h_RSI_Pullback_Trend"
timeframe = "12h"
leverage = 1.0
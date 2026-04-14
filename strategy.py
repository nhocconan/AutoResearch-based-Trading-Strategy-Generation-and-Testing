#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Load weekly data (HTF)
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 50:
        return np.zeros(n)
    
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    volume_1w = df_1w['volume'].values
    
    # Calculate 50-period EMA for weekly trend
    if len(close_1w) >= 50:
        ema_50_1w = pd.Series(close_1w).ewm(span=50, adjust=False, min_periods=50).mean().values
    else:
        ema_50_1w = np.full_like(close_1w, np.nan)
    
    ema_50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_50_1w)
    
    # Calculate weekly ATR (14-period)
    tr1 = high_1w - low_1w
    tr2 = np.abs(high_1w - np.roll(close_1w, 1))
    tr3 = np.abs(low_1w - np.roll(close_1w, 1))
    tr_1w = np.maximum(tr1, np.maximum(tr2, tr3))
    tr_1w[0] = tr1[0]
    
    atr_14_1w = np.full_like(close_1w, np.nan)
    if len(close_1w) >= 14:
        atr_14_1w[13] = np.mean(tr_1w[1:15])
        for i in range(15, len(close_1w)):
            atr_14_1w[i] = (atr_14_1w[i-1] * 13 + tr_1w[i]) / 14
    
    atr_14_1w_aligned = align_htf_to_ltf(prices, df_1w, atr_14_1w)
    
    # Calculate weekly RSI (14-period)
    delta = np.diff(close_1w, prepend=close_1w[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    
    avg_gain = np.full_like(close_1w, np.nan)
    avg_loss = np.full_like(close_1w, np.nan)
    if len(close_1w) >= 14:
        avg_gain[13] = np.mean(gain[1:15])
        avg_loss[13] = np.mean(loss[1:15])
        for i in range(15, len(close_1w)):
            avg_gain[i] = (avg_gain[i-1] * 13 + gain[i]) / 14
            avg_loss[i] = (avg_loss[i-1] * 13 + loss[i]) / 14
    
    rsi_14_1w = np.full_like(close_1w, np.nan)
    for i in range(14, len(close_1w)):
        if avg_loss[i] > 0:
            rs = avg_gain[i] / avg_loss[i]
            rsi_14_1w[i] = 100 - (100 / (1 + rs))
        else:
            rsi_14_1w[i] = 100
    
    rsi_14_1w_aligned = align_htf_to_ltf(prices, df_1w, rsi_14_1w)
    
    # Calculate 6-period RSI for entry timing
    delta_6 = np.diff(close, prepend=close[0])
    gain_6 = np.where(delta_6 > 0, delta_6, 0)
    loss_6 = np.where(delta_6 < 0, -delta_6, 0)
    
    avg_gain_6 = np.full_like(close, np.nan)
    avg_loss_6 = np.full_like(close, np.nan)
    if len(close) >= 6:
        avg_gain_6[5] = np.mean(gain_6[1:7])
        avg_loss_6[5] = np.mean(loss_6[1:7])
        for i in range(7, len(close)):
            avg_gain_6[i] = (avg_gain_6[i-1] * 5 + gain_6[i]) / 6
            avg_loss_6[i] = (avg_loss_6[i-1] * 5 + loss_6[i]) / 6
    
    rsi_6 = np.full_like(close, np.nan)
    for i in range(6, len(close)):
        if avg_loss_6[i] > 0:
            rs = avg_gain_6[i] / avg_loss_6[i]
            rsi_6[i] = 100 - (100 / (1 + rs))
        else:
            rsi_6[i] = 100
    
    signals = np.zeros(n)
    position = 0
    position_size = 0.25
    
    for i in range(100, n):
        # Skip if any critical data is NaN
        if (np.isnan(ema_50_1w_aligned[i]) or 
            np.isnan(atr_14_1w_aligned[i]) or 
            np.isnan(rsi_14_1w_aligned[i]) or 
            np.isnan(rsi_6[i])):
            signals[i] = 0.0
            continue
        
        if position == 0:
            # Long: Weekly uptrend (price > EMA50), not overbought (RSI < 60), RSI(6) oversold bounce
            if (close[i] > ema_50_1w_aligned[i] and
                rsi_14_1w_aligned[i] < 60 and
                rsi_6[i] < 30):
                position = 1
                signals[i] = position_size
            # Short: Weekly downtrend (price < EMA50), not oversold (RSI > 40), RSI(6) overbought bounce
            elif (close[i] < ema_50_1w_aligned[i] and
                  rsi_14_1w_aligned[i] > 40 and
                  rsi_6[i] > 70):
                position = -1
                signals[i] = -position_size
            else:
                signals[i] = 0.0
        elif position == 1:
            # Exit: Weekly trend breaks or RSI(6) overbought
            if (close[i] < ema_50_1w_aligned[i] or 
                rsi_14_1w_aligned[i] > 70 or
                rsi_6[i] > 70):
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = position_size
        elif position == -1:
            # Exit: Weekly trend breaks or RSI(6) oversold
            if (close[i] > ema_50_1w_aligned[i] or 
                rsi_14_1w_aligned[i] < 30 or
                rsi_6[i] < 30):
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -position_size
    
    return signals

name = "6h_1w_EMA50_RSI14_RSI6"
timeframe = "6h"
leverage = 1.0
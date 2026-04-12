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
    
    # Get daily data for EMA filter and volatility regime
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    # Calculate daily 30-period EMA for trend filter
    close_1d = df_1d['close'].values
    ema30_1d = pd.Series(close_1d).ewm(span=30, adjust=False, min_periods=30).mean().values
    ema30_1d_aligned = align_htf_to_ltf(prices, df_1d, ema30_1d)
    
    # Calculate daily ATR for volatility regime
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    tr1 = np.abs(high_1d - low_1d)
    tr2 = np.abs(high_1d - np.roll(close_1d, 1))
    tr3 = np.abs(low_1d - np.roll(close_1d, 1))
    tr1[0] = tr2[0] = tr3[0] = np.nan
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    atr1d = np.full(len(df_1d), np.nan)
    for i in range(14, len(df_1d)):
        atr1d[i] = np.nanmean(tr[i-14:i+1])
    
    # ATR ratio: current daily ATR / 20-period average ATR
    atr_mean20 = np.full(len(df_1d), np.nan)
    for i in range(19, len(df_1d)):
        atr_mean20[i] = np.nanmean(atr1d[i-19:i+1])
    atr_ratio = atr1d / atr_mean20
    atr_ratio_aligned = align_htf_to_ltf(prices, df_1d, atr_ratio)
    
    # Calculate 4-period RSI for mean reversion signals
    delta = np.diff(close, prepend=close[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    
    avg_gain = np.full(n, np.nan)
    avg_loss = np.full(n, np.nan)
    for i in range(3, n):
        if i == 3:
            avg_gain[i] = np.mean(gain[0:4])
            avg_loss[i] = np.mean(loss[0:4])
        else:
            avg_gain[i] = (avg_gain[i-1] * 3 + gain[i]) / 4
            avg_loss[i] = (avg_loss[i-1] * 3 + loss[i]) / 4
    
    rs = np.where(avg_loss != 0, avg_gain / avg_loss, 100)
    rsi4 = 100 - (100 / (1 + rs))
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(30, n):
        # Skip if data not ready
        if (np.isnan(ema30_1d_aligned[i]) or np.isnan(atr_ratio_aligned[i]) or 
            np.isnan(rsi4[i])):
            signals[i] = 0.0
            continue
        
        # Volatility filter: elevated volatility (ATR ratio > 1.2)
        vol_filter = atr_ratio_aligned[i] > 1.2
        
        # Trend filter: price relative to daily EMA
        price_above_ema = close[i] > ema30_1d_aligned[i]
        price_below_ema = close[i] < ema30_1d_aligned[i]
        
        # Mean reversion entries: RSI extremes in direction of trend with volatility
        long_entry = (rsi4[i] < 30) and price_above_ema and vol_filter
        short_entry = (rsi4[i] > 70) and price_below_ema and vol_filter
        
        # Exit: RSI returns to neutral zone or volatility drops
        long_exit = (rsi4[i] > 50) or (atr_ratio_aligned[i] < 0.8)
        short_exit = (rsi4[i] < 50) or (atr_ratio_aligned[i] < 0.8)
        
        if long_entry and position != 1:
            position = 1
            signals[i] = 0.25
        elif short_entry and position != -1:
            position = -1
            signals[i] = -0.25
        elif position == 1 and long_exit:
            position = 0
            signals[i] = 0.0
        elif position == -1 and short_exit:
            position = 0
            signals[i] = 0.0
        else:
            # Hold current position
            if position == 1:
                signals[i] = 0.25
            elif position == -1:
                signals[i] = -0.25
            else:
                signals[i] = 0.0
    
    return signals

name = "4h_1d_rsi4_mean_reversion_vol_filter_v1"
timeframe = "4h"
leverage = 1.0
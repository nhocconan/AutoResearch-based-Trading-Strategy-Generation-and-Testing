#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Load 1d data for HTF indicators - ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    # Calculate daily RSI(14)
    close_1d = df_1d['close'].values
    delta = np.diff(close_1d, prepend=close_1d[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    avg_gain = pd.Series(gain).ewm(alpha=1/14, adjust=False).mean()
    avg_loss = pd.Series(loss).ewm(alpha=1/14, adjust=False).mean()
    rs = avg_gain / (avg_loss + 1e-10)
    rsi_1d = (100 - (100 / (1 + rs))).values
    
    # Calculate daily ADX(14)
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # True Range
    tr1 = high_1d - low_1d
    tr2 = np.abs(high_1d - np.roll(close_1d, 1))
    tr3 = np.abs(low_1d - np.roll(close_1d, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]  # First value
    
    # Directional Movement
    up_move = high_1d - np.roll(high_1d, 1)
    down_move = np.roll(low_1d, 1) - low_1d
    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0)
    
    # Smoothed values
    atr = pd.Series(tr).ewm(alpha=1/14, adjust=False).mean().values
    plus_di = 100 * pd.Series(plus_dm).ewm(alpha=1/14, adjust=False).mean().values / (atr + 1e-10)
    minus_di = 100 * pd.Series(minus_dm).ewm(alpha=1/14, adjust=False).mean().values / (atr + 1e-10)
    dx = 100 * np.abs(plus_di - minus_di) / (plus_di + minus_di + 1e-10)
    adx = pd.Series(dx).ewm(alpha=1/14, adjust=False).mean().values
    
    # Calculate daily 60-period EMA for trend filter
    ema_60_1d = pd.Series(close_1d).ewm(span=60, adjust=False, min_periods=60).mean().values
    
    # Calculate 6-period EMA for 6h timeframe (fast EMA)
    ema_6_6h = pd.Series(close).ewm(span=6, adjust=False, min_periods=6).mean().values
    
    # Calculate 6h volume average (20-period)
    vol_avg_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    # Align all indicators to 6h timeframe
    rsi_1d_aligned = align_htf_to_ltf(prices, df_1d, rsi_1d)
    adx_1d_aligned = align_htf_to_ltf(prices, df_1d, adx)
    ema_60_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_60_1d)
    vol_avg_20_aligned = align_htf_to_ltf(prices, df_1d, vol_avg_20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(1, n):
        # Skip if data not ready
        if (np.isnan(rsi_1d_aligned[i]) or np.isnan(adx_1d_aligned[i]) or 
            np.isnan(ema_60_1d_aligned[i]) or np.isnan(vol_avg_20_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: RSI oversold (<30), strong trend (ADX>25), price above 60-day EMA (uptrend), and volume spike
            if (rsi_1d_aligned[i] < 30 and 
                adx_1d_aligned[i] > 25 and 
                close[i] > ema_60_1d_aligned[i] and 
                volume[i] > 1.5 * vol_avg_20_aligned[i]):
                signals[i] = 0.25
                position = 1
            # Short: RSI overbought (>70), strong trend (ADX>25), price below 60-day EMA (downtrend), and volume spike
            elif (rsi_1d_aligned[i] > 70 and 
                  adx_1d_aligned[i] > 25 and 
                  close[i] < ema_60_1d_aligned[i] and 
                  volume[i] > 1.5 * vol_avg_20_aligned[i]):
                signals[i] = -0.25
                position = -1
        else:
            # Exit: RSI returns to neutral zone (40-60) or trend weakens (ADX<20)
            if position == 1:
                if rsi_1d_aligned[i] > 40 and rsi_1d_aligned[i] < 60:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = 0.25
            else:  # position == -1
                if rsi_1d_aligned[i] > 40 and rsi_1d_aligned[i] < 60:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = -0.25
    
    return signals

name = "6H_RSI14_ADX25_EMA60_Trend_Volume"
timeframe = "6h"
leverage = 1.0
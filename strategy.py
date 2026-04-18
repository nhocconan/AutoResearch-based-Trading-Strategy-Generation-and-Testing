#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Get daily data for calculations
    df_1d = get_htf_data(prices, '1d')
    
    close_1d = df_1d['close'].values
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    volume_1d = df_1d['volume'].values
    
    # Calculate daily RSI(14) for trend filter
    delta = np.diff(close_1d, prepend=close_1d[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    avg_gain = pd.Series(gain).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    avg_loss = pd.Series(loss).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    rs = avg_gain / (avg_loss + 1e-10)
    rsi_1d = 100 - (100 / (1 + rs))
    
    # Calculate daily ATR(14) for volatility filter
    tr1 = high_1d - low_1d
    tr2 = np.abs(high_1d - np.roll(close_1d, 1))
    tr3 = np.abs(low_1d - np.roll(close_1d, 1))
    tr1[0] = high_1d[0] - low_1d[0]
    tr2[0] = 0
    tr3[0] = 0
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    atr_1d = pd.Series(tr).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    
    # Calculate daily EMA(50) for trend confirmation
    ema_50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Calculate 4-day high/low for breakout levels (using daily data)
    high_4d = pd.Series(high_1d).rolling(window=4, min_periods=4).max().values
    low_4d = pd.Series(low_1d).rolling(window=4, min_periods=4).min().values
    
    # Align all daily data to 12h timeframe
    rsi_1d_aligned = align_htf_to_ltf(prices, df_1d, rsi_1d)
    atr_1d_aligned = align_htf_to_ltf(prices, df_1d, atr_1d)
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    high_4d_aligned = align_htf_to_ltf(prices, df_1d, high_4d)
    low_4d_aligned = align_htf_to_ltf(prices, df_1d, low_4d)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 50  # need enough for EMA50
    
    for i in range(start_idx, n):
        # Skip if any required data is not available
        if (np.isnan(rsi_1d_aligned[i]) or np.isnan(atr_1d_aligned[i]) or 
            np.isnan(ema_50_1d_aligned[i]) or np.isnan(high_4d_aligned[i]) or
            np.isnan(low_4d_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Trend filter: RSI between 40 and 60 (avoid extremes)
        rsi_mid = (rsi_1d_aligned[i] >= 40) and (rsi_1d_aligned[i] <= 60)
        
        # Volatility filter: current volatility not too high
        vol_filter = atr_1d_aligned[i] < np.nanmedian(atr_1d_aligned[max(0, i-50):i+1]) * 1.5
        
        # Trend confirmation: price above/below EMA50
        price_above_ema = close[i] > ema_50_1d_aligned[i]
        price_below_ema = close[i] < ema_50_1d_aligned[i]
        
        # Breakout conditions: 4-day high/low breakout
        breakout_up = close[i] > high_4d_aligned[i]
        breakdown_down = close[i] < low_4d_aligned[i]
        
        if position == 0:
            # Long: RSI in middle range + price above EMA50 + breakout up
            if rsi_mid and vol_filter and price_above_ema and breakout_up:
                signals[i] = 0.25
                position = 1
            # Short: RSI in middle range + price below EMA50 + breakdown down
            elif rsi_mid and vol_filter and price_below_ema and breakdown_down:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Long exit: RSI overbought, or price below EMA50, or breakdown
            if (rsi_1d_aligned[i] > 70) or (not price_above_ema) or breakdown_down:
                signals[i] = -0.25  # reverse to short
                position = -1
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: RSI oversold, or price above EMA50, or breakout
            if (rsi_1d_aligned[i] < 30) or (not price_below_ema) or breakout_up:
                signals[i] = 0.25  # reverse to long
                position = 1
            else:
                signals[i] = -0.25
    
    return signals

name = "12h_RSI_EMA_Breakout_VolumeFilter"
timeframe = "12h"
leverage = 1.0
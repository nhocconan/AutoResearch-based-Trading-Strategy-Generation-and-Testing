#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "4h_1d_RSI_Pullback_Trend_V1"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1d data for trend and RSI (once before loop)
    df_1d = get_htf_data(prices, '1d')
    close_1d = df_1d['close'].values
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    
    # Calculate 1d EMA200 for trend
    ema200_1d = pd.Series(close_1d).ewm(span=200, min_periods=200, adjust=False).mean().values
    ema200_1d_aligned = align_htf_to_ltf(prices, df_1d, ema200_1d)
    
    # Calculate 1d RSI(14)
    delta = pd.Series(close_1d).diff()
    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)
    avg_gain = gain.ewm(alpha=1/14, min_periods=14, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1/14, min_periods=14, adjust=False).mean()
    rs = avg_gain / avg_loss
    rsi_1d = 100 - (100 / (1 + rs))
    rsi_1d_values = rsi_1d.fillna(0).values
    rsi_1d_aligned = align_htf_to_ltf(prices, df_1d, rsi_1d_values)
    
    # Calculate 4h RSI(14) for entry timing
    delta_4h = pd.Series(close).diff()
    gain_4h = delta_4h.where(delta_4h > 0, 0)
    loss_4h = -delta_4h.where(delta_4h < 0, 0)
    avg_gain_4h = gain_4h.ewm(alpha=1/14, min_periods=14, adjust=False).mean()
    avg_loss_4h = loss_4h.ewm(alpha=1/14, min_periods=14, adjust=False).mean()
    rs_4h = avg_gain_4h / avg_loss_4h
    rsi_4h = 100 - (100 / (1 + rs_4h))
    rsi_4h_values = rsi_4h.fillna(0).values
    
    # Volume confirmation: current volume > 1.5x 20-period average
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 100
    
    for i in range(start_idx, n):
        if np.isnan(ema200_1d_aligned[i]) or np.isnan(rsi_1d_aligned[i]) or \
           np.isnan(rsi_4h[i]) or np.isnan(vol_ma_20[i]):
            signals[i] = 0.0
            continue
        
        price = close[i]
        vol = volume[i]
        vol_ma = vol_ma_20[i]
        volume_confirmed = vol > 1.5 * vol_ma
        
        if position == 0:
            # Long: Uptrend (price > EMA200) + RSI pullback (RSI < 40) + volume
            if price > ema200_1d_aligned[i] and rsi_1d_aligned[i] < 40 and rsi_4h[i] < 35 and volume_confirmed:
                signals[i] = 0.25
                position = 1
            # Short: Downtrend (price < EMA200) + RSI bounce (RSI > 60) + volume
            elif price < ema200_1d_aligned[i] and rsi_1d_aligned[i] > 60 and rsi_4h[i] > 65 and volume_confirmed:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit: Trend reversal or overbought
            if price < ema200_1d_aligned[i] or rsi_1d_aligned[i] > 70:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit: Trend reversal or oversold
            if price > ema200_1d_aligned[i] or rsi_1d_aligned[i] < 30:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals
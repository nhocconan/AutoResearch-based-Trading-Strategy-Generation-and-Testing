#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "6h_RSI2_Bollinger_Bounce_Volume"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1d data once before loop
    df_1d = get_htf_data(prices, '1d')
    close_1d = df_1d['close'].values
    
    # Calculate 14-period RSI on daily
    delta = np.diff(close_1d, prepend=close_1d[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    
    # Wilder's smoothing
    avg_gain = np.zeros_like(gain)
    avg_loss = np.zeros_like(loss)
    avg_gain[13] = np.mean(gain[1:14])
    avg_loss[13] = np.mean(loss[1:14])
    
    for i in range(14, len(gain)):
        avg_gain[i] = (avg_gain[i-1] * 13 + gain[i]) / 14
        avg_loss[i] = (avg_loss[i-1] * 13 + loss[i]) / 14
    
    rs = np.where(avg_loss != 0, avg_gain / avg_loss, 100)
    rsi_14 = 100 - (100 / (1 + rs))
    rsi_14[:13] = np.nan
    
    # Calculate Bollinger Bands on daily (20, 2)
    sma_20 = pd.Series(close_1d).rolling(window=20, min_periods=20).mean().values
    std_20 = pd.Series(close_1d).rolling(window=20, min_periods=20).std().values
    upper_bb = sma_20 + 2 * std_20
    lower_bb = sma_20 - 2 * std_20
    
    # RSI(2) on 6h for short-term mean reversion
    delta_6h = np.diff(close, prepend=close[0])
    gain_6h = np.where(delta_6h > 0, delta_6h, 0)
    loss_6h = np.where(delta_6h < 0, -delta_6h, 0)
    
    avg_gain_6h = np.zeros_like(gain_6h)
    avg_loss_6h = np.zeros_like(loss_6h)
    for i in range(1, len(gain_6h)):
        avg_gain_6h[i] = (avg_gain_6h[i-1] * 1 + gain_6h[i]) / 2
        avg_loss_6h[i] = (avg_loss_6h[i-1] * 1 + loss_6h[i]) / 2
    
    rs_6h = np.where(avg_loss_6h != 0, avg_gain_6h / avg_loss_6h, 100)
    rsi_2 = 100 - (100 / (1 + rs_6h))
    rsi_2[:1] = np.nan
    
    # Volume confirmation: current volume > 1.5x 20-period average
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    # Align daily indicators to 6h
    rsi_14_6h = align_htf_to_ltf(prices, df_1d, rsi_14)
    sma_20_6h = align_htf_to_ltf(prices, df_1d, sma_20)
    upper_bb_6h = align_htf_to_ltf(prices, df_1d, upper_bb)
    lower_bb_6h = align_htf_to_ltf(prices, df_1d, lower_bb)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 50
    
    for i in range(start_idx, n):
        if np.isnan(rsi_14_6h[i]) or np.isnan(sma_20_6h[i]) or np.isnan(upper_bb_6h[i]) or \
           np.isnan(lower_bb_6h[i]) or np.isnan(rsi_2[i]) or np.isnan(vol_ma_20[i]):
            signals[i] = 0.0
            continue
        
        price = close[i]
        vol = volume[i]
        vol_ma = vol_ma_20[i]
        
        # Volume filter
        volume_ok = vol > 1.5 * vol_ma
        
        if position == 0:
            # Long: RSI2 < 10 (oversold) + price near lower BB + daily RSI < 40 (not overbought)
            if (rsi_2[i] < 10 and 
                price <= lower_bb_6h[i] * 1.02 and  # Allow small tolerance
                rsi_14_6h[i] < 40 and
                volume_ok):
                signals[i] = 0.25
                position = 1
            # Short: RSI2 > 90 (overbought) + price near upper BB + daily RSI > 60 (not oversold)
            elif (rsi_2[i] > 90 and 
                  price >= upper_bb_6h[i] * 0.98 and  # Allow small tolerance
                  rsi_14_6h[i] > 60 and
                  volume_ok):
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit: Price crosses above SMA20 or RSI2 > 50
            if price > sma_20_6h[i] or rsi_2[i] > 50:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit: Price crosses below SMA20 or RSI2 < 50
            if price < sma_20_6h[i] or rsi_2[i] < 50:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals
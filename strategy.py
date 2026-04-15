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
    
    # Get 1d HTF data once before loop
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # Calculate 1d EMA(50) for trend filter
    ema_50_1d = pd.Series(df_1d['close'].values).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # Calculate 1d RSI(14) for overbought/oversold filter
    delta = pd.Series(df_1d['close'].values).diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(alpha=1/14, adjust=False, min_periods=14).mean()
    avg_loss = loss.ewm(alpha=1/14, adjust=False, min_periods=14).mean()
    rs = avg_gain / (avg_loss + 1e-10)
    rsi_14_1d = (100 - (100 / (1 + rs))).values
    rsi_14_1d_aligned = align_htf_to_ltf(prices, df_1d, rsi_14_1d)
    
    # Calculate 4h Donchian(20) channels for entry
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 20:
        return np.zeros(n)
    donchian_high_20_4h = pd.Series(df_4h['high'].values).rolling(window=20, min_periods=20).max().values
    donchian_low_20_4h = pd.Series(df_4h['low'].values).rolling(window=20, min_periods=20).min().values
    donchian_high_20_4h_aligned = align_htf_to_ltf(prices, df_4h, donchian_high_20_4h)
    donchian_low_20_4h_aligned = align_htf_to_ltf(prices, df_4h, donchian_low_20_4h)
    
    # Calculate 4h volume ratio (current vs 20-period average)
    vol_ma_20_4h = pd.Series(df_4h['volume'].values).rolling(window=20, min_periods=20).mean().values
    vol_ma_20_4h_aligned = align_htf_to_ltf(prices, df_4h, vol_ma_20_4h)
    volume_ratio = df_4h['volume'].values / (vol_ma_20_4h_aligned + 1e-10)
    
    signals = np.zeros(n)
    
    for i in range(100, n):
        # Skip if any required data is NaN
        if (np.isnan(ema_50_1d_aligned[i]) or np.isnan(rsi_14_1d_aligned[i]) or 
            np.isnan(donchian_high_20_4h_aligned[i]) or np.isnan(donchian_low_20_4h_aligned[i]) or 
            np.isnan(volume_ratio[i])):
            signals[i] = 0.0
            continue
        
        # Trend filter: 1d EMA50 direction
        trend_up = close[i] > ema_50_1d_aligned[i]
        
        # RSI filter: avoid extreme overbought/oversold
        rsi_ok = (rsi_14_1d_aligned[i] > 30) & (rsi_14_1d_aligned[i] < 70)
        
        # Volume confirmation: volume > 1.5x average
        vol_confirm = volume_ratio[i] > 1.5
        
        # Long conditions: bullish trend + breakout above 4h Donchian high + volume + RSI OK
        if (trend_up and rsi_ok and vol_confirm and 
            close[i] > donchian_high_20_4h_aligned[i]):
            signals[i] = 0.25
            
        # Short conditions: bearish trend + breakdown below 4h Donchian low + volume + RSI OK
        elif ((not trend_up) and rsi_ok and vol_confirm and 
              close[i] < donchian_low_20_4h_aligned[i]):
            signals[i] = -0.25
        else:
            signals[i] = 0.0
    
    return signals

name = "4h_Donchian20_1dEMA50_RSI_Volume_Filter_v1"
timeframe = "4h"
leverage = 1.0
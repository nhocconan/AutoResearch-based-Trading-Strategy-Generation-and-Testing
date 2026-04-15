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
    
    # Get 1d HTF data once before loop
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    # Calculate 1d EMA(21) for trend filter
    ema_21_1d = pd.Series(df_1d['close'].values).ewm(span=21, adjust=False, min_periods=21).mean().values
    ema_21_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_21_1d)
    
    # Calculate 1d RSI(14) for momentum filter
    delta = pd.Series(df_1d['close'].values).diff()
    gain = delta.where(delta > 0, 0.0)
    loss = -delta.where(delta < 0, 0.0)
    avg_gain = pd.Series(gain).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    avg_loss = pd.Series(loss).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    rs = avg_gain / (avg_loss + 1e-10)
    rsi_14_1d = 100 - (100 / (1 + rs))
    rsi_14_1d_aligned = align_htf_to_ltf(prices, df_1d, rsi_14_1d)
    
    # Calculate 6h Donchian(20) channels for breakout signals
    donchian_high_20 = pd.Series(high).rolling(window=20, min_periods=20).max().values
    donchian_low_20 = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # Calculate 6h volume ratio (current vs 20-period average)
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_ratio = volume / (vol_ma_20 + 1e-10)
    
    signals = np.zeros(n)
    
    for i in range(50, n):
        # Skip if any required data is NaN
        if (np.isnan(ema_21_1d_aligned[i]) or np.isnan(rsi_14_1d_aligned[i]) or 
            np.isnan(donchian_high_20[i]) or np.isnan(donchian_low_20[i]) or 
            np.isnan(volume_ratio[i])):
            signals[i] = 0.0
            continue
        
        # Trend filter: 1d EMA21 direction
        bullish_trend = close[i] > ema_21_1d_aligned[i]
        bearish_trend = close[i] < ema_21_1d_aligned[i]
        
        # Momentum filter: 1d RSI in healthy range (not extreme)
        rsi_momentum = (rsi_14_1d_aligned[i] > 30) & (rsi_14_1d_aligned[i] < 70)
        
        # Long conditions:
        # 1. 1d bullish trend (price above EMA21)
        # 2. 1d RSI not overbought (healthy momentum)
        # 3. Price breaks above 6h Donchian(20) high with volume
        # 4. Volume confirmation: volume > 1.5x average
        if (bullish_trend and rsi_momentum and
            close[i] > donchian_high_20[i] and
            volume_ratio[i] > 1.5):
            signals[i] = 0.25
            
        # Short conditions:
        # 1. 1d bearish trend (price below EMA21)
        # 2. 1d RSI not oversold (healthy momentum)
        # 3. Price breaks below 6h Donchian(20) low with volume
        # 4. Volume confirmation: volume > 1.5x average
        elif (bearish_trend and rsi_momentum and
              close[i] < donchian_low_20[i] and
              volume_ratio[i] > 1.5):
            signals[i] = -0.25
        else:
            signals[i] = 0.0
    
    return signals

name = "6h_EMA21_RSI14_Donchian20_Breakout_v1"
timeframe = "6h"
leverage = 1.0
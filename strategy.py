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
    
    # Get 12h HTF data once before loop
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 30:
        return np.zeros(n)
    
    # Calculate 12h Donchian(20) breakout levels
    donchian_high_20 = pd.Series(df_12h['high']).rolling(window=20, min_periods=20).max().values
    donchian_low_20 = pd.Series(df_12h['low']).rolling(window=20, min_periods=20).min().values
    
    # Align 12h Donchian levels to primary timeframe
    donchian_high_20_aligned = align_htf_to_ltf(prices, df_12h, donchian_high_20)
    donchian_low_20_aligned = align_htf_to_ltf(prices, df_12h, donchian_low_20)
    
    # Get 1w HTF data once before loop for trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 50:
        return np.zeros(n)
    
    # Calculate 1w EMA(50) for long-term trend
    ema_50_1w = pd.Series(df_1w['close']).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_50_1w)
    
    # Calculate 12h ATR(14) for volatility filter
    tr1_12h = df_12h['high'] - df_12h['low']
    tr2_12h = np.abs(df_12h['high'] - np.concatenate([[df_12h['close'].iloc[0]], df_12h['close'].iloc[:-1]]))
    tr3_12h = np.abs(df_12h['low'] - np.concatenate([[df_12h['close'].iloc[0]], df_12h['close'].iloc[:-1]]))
    tr_12h = np.maximum(tr1_12h, np.maximum(tr2_12h, tr3_12h))
    atr_14_12h = pd.Series(tr_12h).ewm(span=14, adjust=False, min_periods=14).mean().values
    atr_14_12h_aligned = align_htf_to_ltf(prices, df_12h, atr_14_12h)
    
    # Calculate 12h volume ratio (current vs 20-period average)
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_ratio = volume / (vol_ma_20 + 1e-10)
    
    signals = np.zeros(n)
    
    for i in range(100, n):
        # Skip if any required data is NaN
        if (np.isnan(donchian_high_20_aligned[i]) or np.isnan(donchian_low_20_aligned[i]) or
            np.isnan(ema_50_1w_aligned[i]) or np.isnan(atr_14_12h_aligned[i]) or np.isnan(volume_ratio[i])):
            signals[i] = 0.0
            continue
        
        # Volatility filter: only trade when 12h ATR is elevated (> 0.5% of price)
        vol_filter = atr_14_12h_aligned[i] > 0.005 * close[i]
        
        # Volume confirmation: volume > 1.5x average
        vol_confirm = volume_ratio[i] > 1.5
        
        # Long conditions:
        # 1. Price breaks above 12h Donchian high(20)
        # 2. Price above 1w EMA(50) (bullish long-term trend)
        # 3. Volume confirmation
        # 4. Volatility filter
        if (close[i] > donchian_high_20_aligned[i] and
            close[i] > ema_50_1w_aligned[i] and
            vol_confirm and
            vol_filter):
            signals[i] = 0.25
            
        # Short conditions:
        # 1. Price breaks below 12h Donchian low(20)
        # 2. Price below 1w EMA(50) (bearish long-term trend)
        # 3. Volume confirmation
        # 4. Volatility filter
        elif (close[i] < donchian_low_20_aligned[i] and
              close[i] < ema_50_1w_aligned[i] and
              vol_confirm and
              vol_filter):
            signals[i] = -0.25
        else:
            signals[i] = 0.0
    
    return signals

name = "12h_Donchian20_1w_EMA50_Volume_Filter_v1"
timeframe = "12h"
leverage = 1.0
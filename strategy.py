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
    
    # Get 1d HTF data once before loop (daily trend)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    
    # Calculate 1d EMA50 for trend filter
    ema_50_1d = pd.Series(df_1d['close'].values).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # Get 1d HTF data for daily Donchian channel (20-period)
    # Donchian upper = max(high, 20), Donchian lower = min(low, 20)
    donchian_upper = pd.Series(df_1d['high'].values).rolling(window=20, min_periods=20).max().values
    donchian_lower = pd.Series(df_1d['low'].values).rolling(window=20, min_periods=20).min().values
    
    # Align Donchian levels to 12h
    donchian_upper_12h = align_htf_to_ltf(prices, df_1d, donchian_upper)
    donchian_lower_12h = align_htf_to_ltf(prices, df_1d, donchian_lower)
    
    # Calculate 12h ATR(14) for volatility filter and stoploss
    tr1 = high - low
    tr2 = np.abs(high - np.concatenate([[close[0]], close[:-1]]))
    tr3 = np.abs(low - np.concatenate([[close[0]], close[:-1]]))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    atr_14 = pd.Series(tr).ewm(span=14, adjust=False, min_periods=14).mean().values
    
    # Calculate 12h volume ratio (current vs 20-period average)
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_ratio = volume / (vol_ma_20 + 1e-10)
    
    signals = np.zeros(n)
    
    for i in range(50, n):
        # Skip if any required data is NaN
        if (np.isnan(ema_50_1d_aligned[i]) or np.isnan(donchian_upper_12h[i]) or 
            np.isnan(donchian_lower_12h[i]) or np.isnan(atr_14[i]) or np.isnan(volume_ratio[i])):
            signals[i] = 0.0
            continue
        
        # Long conditions:
        # 1. Daily trend filter: price above 1d EMA50 (bullish daily bias)
        # 2. Price breaks above 1d Donchian upper with volume confirmation
        # 3. Volume confirmation: volume > 1.3x average
        # 4. Volatility filter: ATR > 0.4% of price (avoid extremely low volatility)
        if (close[i] > ema_50_1d_aligned[i] and
            close[i] > donchian_upper_12h[i] and
            volume_ratio[i] > 1.3 and
            atr_14[i] > 0.004 * close[i]):
            signals[i] = 0.25
            
        # Short conditions:
        # 1. Daily trend filter: price below 1d EMA50 (bearish daily bias)
        # 2. Price breaks below 1d Donchian lower with volume confirmation
        # 3. Volume confirmation: volume > 1.3x average
        # 4. Volatility filter: ATR > 0.4% of price
        elif (close[i] < ema_50_1d_aligned[i] and
              close[i] < donchian_lower_12h[i] and
              volume_ratio[i] > 1.3 and
              atr_14[i] > 0.004 * close[i]):
            signals[i] = -0.25
        else:
            signals[i] = 0.0
    
    return signals

name = "12h_1d_EMA50_Donchian20_Breakout_Volume_Filter_v1"
timeframe = "12h"
leverage = 1.0
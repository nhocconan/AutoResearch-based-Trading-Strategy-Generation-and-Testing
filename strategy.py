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
    
    # Get 1w HTF data once before loop (weekly trend)
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 20:
        return np.zeros(n)
    
    # Calculate 1w EMA20 for trend filter
    ema_20_1w = pd.Series(df_1w['close'].values).ewm(span=20, adjust=False, min_periods=20).mean().values
    ema_20_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_20_1w)
    
    # Get 1d HTF data for daily Donchian channels
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    
    # Calculate daily Donchian(20) channels (using prior day's data)
    donchian_high_20 = pd.Series(df_1d['high'].values).rolling(window=20, min_periods=20).max().shift(1).values
    donchian_low_20 = pd.Series(df_1d['low'].values).rolling(window=20, min_periods=20).min().shift(1).values
    
    # Align Donchian levels to 1d
    donchian_high_20_aligned = align_htf_to_ltf(prices, df_1d, donchian_high_20)
    donchian_low_20_aligned = align_htf_to_ltf(prices, df_1d, donchian_low_20)
    
    # Calculate 1d ATR(14) for volatility filter
    tr1 = high - low
    tr2 = np.abs(high - np.concatenate([[close[0]], close[:-1]]))
    tr3 = np.abs(low - np.concatenate([[close[0]], close[:-1]]))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    atr_14 = pd.Series(tr).ewm(span=14, adjust=False, min_periods=14).mean().values
    
    # Calculate 1d volume ratio (current vs 20-period average)
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_ratio = volume / (vol_ma_20 + 1e-10)
    
    signals = np.zeros(n)
    
    for i in range(100, n):
        # Skip if any required data is NaN
        if (np.isnan(ema_20_1w_aligned[i]) or np.isnan(donchian_high_20_aligned[i]) or 
            np.isnan(donchian_low_20_aligned[i]) or np.isnan(atr_14[i]) or np.isnan(volume_ratio[i])):
            signals[i] = 0.0
            continue
        
        # Long conditions:
        # 1. Weekly trend filter: price above 1w EMA20 (bullish weekly bias)
        # 2. Price breaks above daily Donchian(20) high with volume (bullish breakout)
        # 3. Volume confirmation: volume > 1.5x average
        # 4. Volatility filter: ATR > 0.3% of price (avoid extremely low volatility)
        if (close[i] > ema_20_1w_aligned[i] and
            close[i] > donchian_high_20_aligned[i] and
            volume_ratio[i] > 1.5 and
            atr_14[i] > 0.003 * close[i]):
            signals[i] = 0.25
            
        # Short conditions:
        # 1. Weekly trend filter: price below 1w EMA20 (bearish weekly bias)
        # 2. Price breaks below daily Donchian(20) low with volume (bearish breakdown)
        # 3. Volume confirmation: volume > 1.5x average
        # 4. Volatility filter: ATR > 0.3% of price
        elif (close[i] < ema_20_1w_aligned[i] and
              close[i] < donchian_low_20_aligned[i] and
              volume_ratio[i] > 1.5 and
              atr_14[i] > 0.003 * close[i]):
            signals[i] = -0.25
        else:
            signals[i] = 0.0
    
    return signals

name = "1d_1w_EMA20_1d_Donchian20_Breakout_Volume_Filter_v1"
timeframe = "1d"
leverage = 1.0
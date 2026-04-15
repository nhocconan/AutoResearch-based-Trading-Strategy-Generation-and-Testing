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
    
    # Get 1d HTF data once before loop (primary timeframe)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    # Get 1w HTF data for trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 20:
        return np.zeros(n)
    
    # Calculate 1d ATR(14) for volatility filter
    tr1 = high - low
    tr2 = np.abs(high - np.concatenate([[close[0]], close[:-1]]))
    tr3 = np.abs(low - np.concatenate([[close[0]], close[:-1]]))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    atr_14 = pd.Series(tr).ewm(span=14, adjust=False, min_periods=14).mean().values
    
    # Calculate 1d Donchian channels (20-period) from 1d data
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    upper_20_1d = pd.Series(high_1d).rolling(window=20, min_periods=20).max().values
    lower_20_1d = pd.Series(low_1d).rolling(window=20, min_periods=20).min().values
    
    # Align 1d Donchian to 1d (no shift needed as we're on same timeframe)
    upper_20_1d_aligned = upper_20_1d  # Already aligned to 1d bars
    lower_20_1d_aligned = lower_20_1d
    
    # Calculate 1w EMA(50) for trend filter
    close_1w = df_1w['close'].values
    ema_50_1w = pd.Series(close_1w).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_50_1w)
    
    # Calculate 1d volume ratio (current vs 20-period average)
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_ratio = volume / (vol_ma_20 + 1e-10)
    
    signals = np.zeros(n)
    
    # Precompute session filter (00-24 UTC for 1d - always true, kept for structure)
    hours = prices.index.hour
    in_session = (hours >= 0) & (hours <= 23)  # Always true for 1d
    
    for i in range(100, n):
        # Skip if any required data is NaN
        if (np.isnan(upper_20_1d_aligned[i]) or np.isnan(lower_20_1d_aligned[i]) or 
            np.isnan(ema_50_1w_aligned[i]) or np.isnan(atr_14[i]) or 
            np.isnan(volume_ratio[i]) or not in_session[i]):
            signals[i] = 0.0
            continue
        
        # Long conditions:
        # 1. 1d price breaks above 1d Donchian upper (20) - bullish breakout
        # 2. Price above 1w EMA50 (bullish weekly trend)
        # 3. Volume confirmation: volume > 1.5x average
        # 4. Volatility filter: ATR > 0.5% of price (avoid low volatility chop)
        if (close[i] > upper_20_1d_aligned[i] and
            close[i] > ema_50_1w_aligned[i] and
            volume_ratio[i] > 1.5 and
            atr_14[i] > 0.005 * close[i]):
            signals[i] = 0.25
            
        # Short conditions:
        # 1. 1d price breaks below 1d Donchian lower (20) - bearish breakdown
        # 2. Price below 1w EMA50 (bearish weekly trend)
        # 3. Volume confirmation: volume > 1.5x average
        # 4. Volatility filter: ATR > 0.5% of price
        elif (close[i] < lower_20_1d_aligned[i] and
              close[i] < ema_50_1w_aligned[i] and
              volume_ratio[i] > 1.5 and
              atr_14[i] > 0.005 * close[i]):
            signals[i] = -0.25
        else:
            signals[i] = 0.0
    
    return signals

name = "1d_Donchian20_1w_EMA50_Volume_Filter_v1"
timeframe = "1d"
leverage = 1.0
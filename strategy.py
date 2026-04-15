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
    if len(df_1d) < 30:
        return np.zeros(n)
    
    # Calculate 1d Donchian channels (20-period) - primary structure
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    upper_20_1d = pd.Series(high_1d).rolling(window=20, min_periods=20).max().values
    lower_20_1d = pd.Series(low_1d).rolling(window=20, min_periods=20).min().values
    
    # Align 1d Donchian to 1d timeframe (no shift needed, already aligned)
    upper_20_aligned = upper_20_1d  # Already at 1d frequency
    lower_20_aligned = lower_20_1d  # Already at 1d frequency
    
    # Get 1w HTF data for weekly trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 20:
        return np.zeros(n)
    
    # Calculate 1w EMA(21) for weekly trend
    close_1w = df_1w['close'].values
    ema_21_1w = pd.Series(close_1w).ewm(span=21, adjust=False, min_periods=21).mean().values
    
    # Align 1w EMA to 1d timeframe with proper delay (wait for weekly close)
    ema_21_aligned = align_htf_to_ltf(prices, df_1w, ema_21_1w)
    
    # Calculate 1d ATR(14) for volatility filter and stoploss
    tr1 = high - low
    tr2 = np.abs(high - np.concatenate([[close[0]], close[:-1]]))
    tr3 = np.abs(low - np.concatenate([[close[0]], close[:-1]]))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    atr_14 = pd.Series(tr).ewm(span=14, adjust=False, min_periods=14).mean().values
    
    # Calculate 1d volume ratio (current vs 20-period average)
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_ratio = volume / (vol_ma_20 + 1e-10)
    
    signals = np.zeros(n)
    position = 0  # Track current position: 1=long, -1=short, 0=flat
    entry_price = 0.0
    
    # Warmup period: need enough data for all indicators
    warmup = max(100, 20, 21)  # 20 for Donchian, 21 for EMA
    
    for i in range(warmup, n):
        # Skip if any required data is NaN
        if (np.isnan(upper_20_aligned[i]) or np.isnan(lower_20_aligned[i]) or 
            np.isnan(ema_21_aligned[i]) or np.isnan(atr_14[i]) or 
            np.isnan(volume_ratio[i])):
            signals[i] = 0.0
            continue
        
        # Get current values
        curr_close = close[i]
        curr_upper = upper_20_aligned[i]
        curr_lower = lower_20_aligned[i]
        curr_ema = ema_21_aligned[i]
        curr_atr = atr_14[i]
        curr_vol_ratio = volume_ratio[i]
        
        # Stoploss: exit if price moves against position by 2.0 * ATR
        if position == 1 and curr_close < entry_price - 2.0 * curr_atr:
            signals[i] = 0.0
            position = 0
            continue
        elif position == -1 and curr_close > entry_price + 2.0 * curr_atr:
            signals[i] = 0.0
            position = 0
            continue
        
        # Long conditions:
        # 1. Price breaks above 1d Donchian upper (20) - bullish breakout
        # 2. Weekly EMA(21) filter: price above weekly EMA (bullish weekly trend)
        # 3. Volume confirmation: volume > 1.5x average
        if (curr_close > curr_upper and
            curr_close > curr_ema and
            curr_vol_ratio > 1.5 and
            position != 1):
            signals[i] = 0.25
            position = 1
            entry_price = curr_close
            
        # Short conditions:
        # 1. Price breaks below 1d Donchian lower (20) - bearish breakdown
        # 2. Weekly EMA(21) filter: price below weekly EMA (bearish weekly trend)
        # 3. Volume confirmation: volume > 1.5x average
        elif (curr_close < curr_lower and
              curr_close < curr_ema and
              curr_vol_ratio > 1.5 and
              position != -1):
            signals[i] = -0.25
            position = -1
            entry_price = curr_close
            
        # Hold current position
        else:
            signals[i] = float(position) * 0.25 if position != 0 else 0.0
    
    return signals

name = "1d_Donchian20_1w_EMA21_Volume_Filter_v1"
timeframe = "1d"
leverage = 1.0
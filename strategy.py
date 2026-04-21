#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Load daily data ONCE before loop for trend, volatility, and volume context
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # Daily EMA50 for trend filter
    close_1d = df_1d['close'].values
    ema_50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # Daily ATR for volatility regime filter
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    tr1 = high_1d - low_1d
    tr2 = np.abs(high_1d - np.roll(close_1d, 1))
    tr3 = np.abs(low_1d - np.roll(close_1d, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]
    atr_14 = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    # Daily volatility ratio: current ATR / 50-period average ATR
    atr_ma_50 = pd.Series(atr_14).rolling(window=50, min_periods=50).mean().values
    atr_ratio = atr_14 / atr_ma_50
    atr_ratio_aligned = align_htf_to_ltf(prices, df_1d, atr_ratio)
    
    # Daily volume ratio: volume / 20-day average volume
    vol_ma_20 = pd.Series(df_1d['volume'].values).rolling(window=20, min_periods=20).mean().values
    vol_ratio_1d = df_1d['volume'].values / vol_ma_20
    vol_ratio_1d_aligned = align_htf_to_ltf(prices, df_1d, vol_ratio_1d)
    
    # 4h Bollinger Bands for entry timing (lower timeframe precision)
    close_4h = prices['close'].values
    sma_20 = pd.Series(close_4h).rolling(window=20, min_periods=20).mean().values
    std_20 = pd.Series(close_4h).rolling(window=20, min_periods=20).std().values
    upper_bb = sma_20 + 2 * std_20
    lower_bb = sma_20 - 2 * std_20
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(50, n):
        # Skip if indicators not ready
        if (np.isnan(ema_50_1d_aligned[i]) or np.isnan(atr_ratio_aligned[i]) or 
            np.isnan(vol_ratio_1d_aligned[i]) or np.isnan(sma_20[i]) or 
            np.isnan(std_20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        price_close = prices['close'].iloc[i]
        daily_ema = ema_50_1d_aligned[i]
        atr_ratio_val = atr_ratio_aligned[i]
        vol_ratio_val = vol_ratio_1d_aligned[i]
        upper_bb_val = upper_bb[i]
        lower_bb_val = lower_bb[i]
        
        if position == 0:
            # Enter long: price above daily EMA, low volatility regime, volume confirmation, near BB lower band
            if (price_close > daily_ema and 
                atr_ratio_val < 1.2 and 
                vol_ratio_val > 1.3 and 
                price_close < lower_bb_val * 1.02):  # slightly below or at lower BB
                signals[i] = 0.25
                position = 1
            # Enter short: price below daily EMA, low volatility regime, volume confirmation, near BB upper band
            elif (price_close < daily_ema and 
                  atr_ratio_val < 1.2 and 
                  vol_ratio_val > 1.3 and 
                  price_close > upper_bb_val * 0.98):  # slightly above or at upper BB
                signals[i] = -0.25
                position = -1
        
        elif position != 0:
            # Exit: reverse crossover or volatility expansion
            if position == 1 and (price_close < daily_ema or atr_ratio_val > 1.8):
                signals[i] = 0.0
                position = 0
            elif position == -1 and (price_close > daily_ema or atr_ratio_val > 1.8):
                signals[i] = 0.0
                position = 0
            else:
                # Hold position
                signals[i] = 0.25 if position == 1 else -0.25
    
    return signals

name = "4h_DailyEMA50_BB_Pullback_Volume_VolatilityFilter"
timeframe = "4h"
leverage = 1.0
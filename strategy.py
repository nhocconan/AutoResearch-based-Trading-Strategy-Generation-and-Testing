#!/usr/bin/env python3
"""
6h_RSI2_Contrast_Stochastic_VolumeFilter
Hypothesis: Contrarian mean reversion on 6h using 2-period RSI (RSI2) combined with Stochastic(14,3,3) for oversold/overbought confirmation, filtered by volume spike and 12h EMA50 trend. 
Enters long when RSI2 < 10, Stochastic %K < 20, volume > 1.5x average, and price > 12h EMA50 (uptrend filter). 
Enters short when RSI2 > 90, Stochastic %K > 80, volume > 1.5x average, and price < 12h EMA50 (downtrend filter). 
Exits on RSI2 > 50 (long) or RSI2 < 50 (short) or opposite Stochastic extreme. 
Designed for low trade frequency (target: 12-25 trades/year) to minimize fee drift. 
Works in bull/bear via 12h EMA50 trend filter and volume confirmation as regime filter.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Load HTF data ONCE before loop (12h for EMA50 trend filter)
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 50:
        return np.zeros(n)
    
    # === 12h EMA50 for HTF trend filter ===
    close_12h = df_12h['close'].values
    ema_50_12h = pd.Series(close_12h).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_12h_aligned = align_htf_to_ltf(prices, df_12h, ema_50_12h)
    
    # === Volume spike filter (20-period average) ===
    volume = prices['volume'].values
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    # === RSI(2) on 6h close ===
    close = prices['close'].values
    delta = pd.Series(close).diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = pd.Series(gain).ewm(alpha=1/2, adjust=False, min_periods=2).mean().values
    avg_loss = pd.Series(loss).ewm(alpha=1/2, adjust=False, min_periods=2).mean().values
    rs = avg_gain / (avg_loss + 1e-10)
    rsi_2 = 100 - (100 / (1 + rs))
    
    # === Stochastic(14,3,3) on 6h high/low/close ===
    lookback = 14
    highest_high = pd.Series(prices['high'].values).rolling(window=lookback, min_periods=lookback).max().values
    lowest_low = pd.Series(prices['low'].values).rolling(window=lookback, min_periods=lookback).min().values
    stoch_k_raw = 100 * (close - lowest_low) / (highest_high - lowest_low + 1e-10)
    # Smooth %K with 3-period SMA
    stoch_k = pd.Series(stoch_k_raw).rolling(window=3, min_periods=3).mean().values
    # Smooth %D with 3-period SMA (not used directly but for completeness)
    stoch_d = pd.Series(stoch_k).rolling(window=3, min_periods=3).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    for i in range(50, n):
        # Skip if indicators not ready
        if (np.isnan(ema_50_12h_aligned[i]) or np.isnan(vol_ma[i]) 
            or np.isnan(rsi_2[i]) or np.isnan(stoch_k[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        price = close[i]
        
        if position == 0:
            # Volume confirmation: current volume > 1.5x 20-period average
            vol_confirm = volume[i] > 1.5 * vol_ma[i]
            
            # Long conditions: RSI2 < 10 (extreme oversold), Stoch %K < 20, volume spike, 12h uptrend
            long_rsi2 = rsi_2[i] < 10
            long_stoch = stoch_k[i] < 20
            long_trend = price > ema_50_12h_aligned[i]
            
            # Short conditions: RSI2 > 90 (extreme overbought), Stoch %K > 80, volume spike, 12h downtrend
            short_rsi2 = rsi_2[i] > 90
            short_stoch = stoch_k[i] > 80
            short_trend = price < ema_50_12h_aligned[i]
            
            # Entry logic
            if long_rsi2 and long_stoch and vol_confirm and long_trend:
                signals[i] = 0.25
                position = 1
                entry_price = price
            elif short_rsi2 and short_stoch and vol_confirm and short_trend:
                signals[i] = -0.25
                position = -1
                entry_price = price
        
        elif position == 1:
            # Exit conditions: RSI2 > 50 (recovered from oversold) or Stoch %K > 80 (overbought)
            if rsi_2[i] > 50 or stoch_k[i] > 80:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit conditions: RSI2 < 50 (recovered from overbought) or Stoch %K < 20 (oversold)
            if rsi_2[i] < 50 or stoch_k[i] < 20:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "6h_RSI2_Contrast_Stochastic_VolumeFilter"
timeframe = "6h"
leverage = 1.0
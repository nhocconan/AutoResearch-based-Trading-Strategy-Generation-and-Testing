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
    
    # Get daily HTF data once before loop
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    daily_close = df_1d['close'].values
    daily_high = df_1d['high'].values
    daily_low = df_1d['low'].values
    daily_volume = df_1d['volume'].values
    
    # Calculate daily ATR(14) for volatility regime filter
    tr1 = daily_high - daily_low
    tr2 = np.abs(daily_high - np.concatenate([[daily_close[0]], daily_close[:-1]]))
    tr3 = np.abs(daily_low - np.concatenate([[daily_close[0]], daily_close[:-1]]))
    daily_tr = np.maximum(tr1, np.maximum(tr2, tr3))
    daily_atr_14 = pd.Series(daily_tr).ewm(span=14, adjust=False, min_periods=14).mean().values
    
    # Align daily ATR to 12h timeframe
    daily_atr_14_12h = align_htf_to_ltf(prices, df_1d, daily_atr_14)
    
    # Calculate 12h ATR(14) for stoploss reference
    tr1_12h = high - low
    tr2_12h = np.abs(high - np.concatenate([[close[0]], close[:-1]]))
    tr3_12h = np.abs(low - np.concatenate([[close[0]], close[:-1]]))
    tr_12h = np.maximum(tr1_12h, np.maximum(tr2_12h, tr3_12h))
    atr_14_12h = pd.Series(tr_12h).ewm(span=14, adjust=False, min_periods=14).mean().values
    
    # Calculate 12h volume ratio (current vs 20-period average)
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_ratio = volume / (vol_ma_20 + 1e-10)
    
    signals = np.zeros(n)
    
    for i in range(100, n):
        # Skip if any required data is NaN
        if (np.isnan(daily_atr_14_12h[i]) or np.isnan(atr_14_12h[i]) or 
            np.isnan(volume_ratio[i])):
            signals[i] = 0.0
            continue
        
        # Regime filter: Daily ATR > 1.5% of price (avoid low volatility chop)
        volatility_regime = daily_atr_14_12h[i] > 0.015 * close[i]
        
        # Entry conditions with discrete sizing (0.25)
        # Long: 12h close above daily high + volume confirmation + volatility regime
        if (close[i] > daily_high[i] and            # 12h price above daily high (breakout)
            volume_ratio[i] > 1.4 and               # Volume confirmation
            volatility_regime):                     # Volatility regime filter
            signals[i] = 0.25
            
        # Short: 12h close below daily low + volume confirmation + volatility regime
        elif (close[i] < daily_low[i] and           # 12h price below daily low (breakdown)
              volume_ratio[i] > 1.4 and             # Volume confirmation
              volatility_regime):                   # Volatility regime filter
            signals[i] = -0.25
        else:
            signals[i] = 0.0
    
    return signals

name = "12h_Daily_High_Low_Breakout_Volume_Volatility_Regime"
timeframe = "12h"
leverage = 1.0
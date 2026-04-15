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
    
    # Calculate 12h ATR(14) for volatility filter (using 12h data)
    tr1 = high - low
    tr2 = np.abs(high - np.concatenate([[close[0]], close[:-1]]))
    tr3 = np.abs(low - np.concatenate([[close[0]], close[:-1]]))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    atr_14 = pd.Series(tr).ewm(span=14, adjust=False, min_periods=14).mean().values
    
    # Calculate 12h volume ratio (current vs 20-period average)
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_ratio = volume / (vol_ma_20 + 1e-10)
    
    # Calculate daily ATR for volatility regime filter (using daily data)
    daily_tr1 = daily_high - daily_low
    daily_tr2 = np.abs(daily_high - np.concatenate([[daily_close[0]], daily_close[:-1]]))
    daily_tr3 = np.abs(daily_low - np.concatenate([[daily_close[0]], daily_close[:-1]]))
    daily_tr = np.maximum(daily_tr1, np.maximum(daily_tr2, daily_tr3))
    daily_atr_14 = pd.Series(daily_tr).ewm(span=14, adjust=False, min_periods=14).mean().values
    daily_atr_ma_50 = pd.Series(daily_atr_14).rolling(window=50, min_periods=50).mean().values
    daily_atr_ratio = daily_atr_14 / (daily_atr_ma_50 + 1e-10)
    
    # Align HTF daily ATR ratio to 12h timeframe
    daily_atr_ratio_12h = align_htf_to_ltf(prices, df_1d, daily_atr_ratio)
    
    signals = np.zeros(n)
    
    for i in range(100, n):
        # Skip if any required data is NaN
        if (np.isnan(atr_14[i]) or np.isnan(volume_ratio[i]) or 
            np.isnan(daily_atr_ratio_12h[i])):
            signals[i] = 0.0
            continue
        
        # Regime filter: High volatility regime (ATR ratio > 1.2) for breakout validity
        vol_regime = daily_atr_ratio_12h[i] > 1.2
        
        # Entry conditions:
        # 1. 12h price breaks above previous day's high with volume confirmation → long
        # 2. 12h price breaks below previous day's low with volume confirmation → short
        # 3. Volume confirmation: volume > 1.5x average
        # 4. Volatility regime filter: ATR ratio > 1.2 (high volatility environment)
        # 5. Discrete position sizing: 0.25
        
        # Get previous day's high and low (aligned to current 12h bar)
        prev_daily_high = daily_high[:-1]  # yesterday's high
        prev_daily_low = daily_low[:-1]    # yesterday's low
        # Align to 12h timeframe
        prev_daily_high_12h = align_htf_to_ltf(prices, df_1d, prev_daily_high)
        prev_daily_low_12h = align_htf_to_ltf(prices, df_1d, prev_daily_low)
        
        # Long conditions: 12h breakout above previous day's high
        if (close[i] > prev_daily_high_12h[i] and            # 12h price above yesterday's high
            volume_ratio[i] > 1.5 and                        # Volume confirmation
            vol_regime):                                     # High volatility regime
            signals[i] = 0.25
            
        # Short conditions: 12h breakdown below previous day's low
        elif (close[i] < prev_daily_low_12h[i] and           # 12h price below yesterday's low
              volume_ratio[i] > 1.5 and                      # Volume confirmation
              vol_regime):                                   # High volatility regime
            signals[i] = -0.25
        else:
            signals[i] = 0.0
    
    return signals

name = "12h_PreviousDay_HighLow_Breakout_Volume_VolatilityRegime"
timeframe = "12h"
leverage = 1.0
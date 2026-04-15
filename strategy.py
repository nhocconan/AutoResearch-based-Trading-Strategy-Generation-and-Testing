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
    
    # Calculate daily ATR(14) for volatility filter
    tr1 = daily_high - daily_low
    tr2 = np.abs(daily_high - np.concatenate([[daily_close[0]], daily_close[:-1]]))
    tr3 = np.abs(daily_low - np.concatenate([[daily_close[0]], daily_close[:-1]]))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    atr_14_daily = pd.Series(tr).ewm(span=14, adjust=False, min_periods=14).mean().values
    
    # Align HTF ATR to 4h timeframe
    atr_14_4h = align_htf_to_ltf(prices, df_1d, atr_14_daily)
    
    # Calculate 4h ATR(14) for stoploss reference
    tr1_4h = high - low
    tr2_4h = np.abs(high - np.concatenate([[close[0]], close[:-1]]))
    tr3_4h = np.abs(low - np.concatenate([[close[0]], close[:-1]]))
    tr_4h = np.maximum(tr1_4h, np.maximum(tr2_4h, tr3_4h))
    atr_14_4h_calc = pd.Series(tr_4h).ewm(span=14, adjust=False, min_periods=14).mean().values
    
    # Calculate 4h volume ratio (current vs 20-period average)
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_ratio = volume / (vol_ma_20 + 1e-10)
    
    signals = np.zeros(n)
    
    for i in range(100, n):
        # Skip if any required data is NaN
        if (np.isnan(atr_14_4h[i]) or np.isnan(atr_14_4h_calc[i]) or np.isnan(volume_ratio[i])):
            signals[i] = 0.0
            continue
        
        # Entry conditions:
        # 1. 4h price above/below daily ATR bands from previous day close
        # 2. Volume confirmation: volume > 1.5x average
        # 3. Volatility filter: daily ATR > 0.8% of price (avoid low volatility chop)
        # 4. Discrete position sizing: 0.25
        
        # Long conditions: 4h breakout above previous day close + 0.5 * daily ATR
        long_threshold = daily_close[i-1] + 0.5 * atr_14_daily[i-1] if i > 0 and not np.isnan(daily_close[i-1]) and not np.isnan(atr_14_daily[i-1]) else np.nan
        if (not np.isnan(long_threshold) and
            close[i] > long_threshold and            # 4h price above previous day close + 0.5*ATR
            volume_ratio[i] > 1.5 and                # Volume confirmation
            atr_14_4h[i] > 0.008 * close[i]):        # Volatility filter
            signals[i] = 0.25
            
        # Short conditions: 4h breakdown below previous day close - 0.5 * daily ATR
        elif (not np.isnan(long_threshold) and
              close[i] < long_threshold - atr_14_daily[i-1] and  # 4h price below previous day close - 0.5*ATR
              volume_ratio[i] > 1.5 and                # Volume confirmation
              atr_14_4h[i] > 0.008 * close[i]):        # Volatility filter
            signals[i] = -0.25
        else:
            signals[i] = 0.0
    
    return signals

name = "4h_DailyATR_Breakout_Volume_Filter"
timeframe = "4h"
leverage = 1.0
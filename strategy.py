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
    
    # Calculate daily ATR(14) for volatility filter
    tr1 = pd.Series(daily_high - daily_low)
    tr2 = pd.Series(np.abs(daily_high - np.concatenate([[daily_close[0]], daily_close[:-1]])))
    tr3 = pd.Series(np.abs(daily_low - np.concatenate([[daily_close[0]], daily_close[:-1]])))
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr_14 = tr.ewm(span=14, adjust=False, min_periods=14).mean().values
    
    # Calculate daily EMA(20) for trend filter
    ema_20 = pd.Series(daily_close).ewm(span=20, adjust=False, min_periods=20).mean().values
    
    # Align HTF indicators to 6h timeframe with proper delay
    atr_14_6h = align_htf_to_ltf(prices, df_1d, atr_14)
    ema_20_6h = align_htf_to_ltf(prices, df_1d, ema_20)
    
    # Calculate 6h Bollinger Bands (20, 2)
    sma_20 = pd.Series(close).rolling(window=20, min_periods=20).mean().values
    std_20 = pd.Series(close).rolling(window=20, min_periods=20).std().values
    upper_bb = sma_20 + 2 * std_20
    lower_bb = sma_20 - 2 * std_20
    bb_width = (upper_bb - lower_bb) / (sma_20 + 1e-10)
    
    # Calculate 6h volume ratio (current vs 20-period average)
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_ratio = volume / (vol_ma_20 + 1e-10)
    
    signals = np.zeros(n)
    
    for i in range(100, n):
        # Skip if any required data is NaN
        if (np.isnan(atr_14_6h[i]) or np.isnan(ema_20_6h[i]) or np.isnan(upper_bb[i]) or 
            np.isnan(lower_bb[i]) or np.isnan(bb_width[i]) or np.isnan(volume_ratio[i])):
            signals[i] = 0.0
            continue
        
        # Entry conditions:
        # 1. Bollinger Band squeeze: width < 0.05 (5% of price)
        # 2. Breakout above upper BB with volume confirmation
        # 3. Breakdown below lower BB with volume confirmation
        # 4. Daily trend filter: price above/below daily EMA20
        # 5. Daily volatility filter: ATR > 0.5% of price (avoid low volatility chop)
        # 6. Discrete position sizing: 0.25
        
        # Long conditions: BB breakout above with volume in daily uptrend
        if (bb_width[i] < 0.05 and           # Bollinger Band squeeze
            close[i] > upper_bb[i] and       # Breakout above upper BB
            close[i] > ema_20_6h[i] and      # Daily uptrend filter
            volume_ratio[i] > 2.0 and        # Volume confirmation (2x average)
            atr_14_6h[i] > 0.005 * close[i]): # Volatility filter
            signals[i] = 0.25
            
        # Short conditions: BB breakdown below with volume in daily downtrend
        elif (bb_width[i] < 0.05 and         # Bollinger Band squeeze
              close[i] < lower_bb[i] and     # Breakdown below lower BB
              close[i] < ema_20_6h[i] and    # Daily downtrend filter
              volume_ratio[i] > 2.0 and      # Volume confirmation (2x average)
              atr_14_6h[i] > 0.005 * close[i]): # Volatility filter
            signals[i] = -0.25
        else:
            signals[i] = 0.0
    
    return signals

name = "6h_Bollinger_Squeeze_Breakout_Volume_EMA20_ATR_Filter"
timeframe = "6h"
leverage = 1.0
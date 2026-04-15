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
    
    # Calculate daily Williams %R (14-period) - mean reversion indicator
    # %R = (Highest High - Close) / (Highest High - Lowest Low) * -100
    highest_high_14 = pd.Series(daily_high).rolling(window=14, min_periods=14).max().values
    lowest_low_14 = pd.Series(daily_low).rolling(window=14, min_periods=14).min().values
    williams_r = (highest_high_14 - daily_close) / (highest_high_14 - lowest_low_14 + 1e-10) * -100
    
    # Calculate daily ATR(14) for volatility filter
    tr1 = pd.Series(daily_high - daily_low)
    tr2 = pd.Series(np.abs(daily_high - np.concatenate([[daily_close[0]], daily_close[:-1]])))
    tr3 = pd.Series(np.abs(daily_low - np.concatenate([[daily_close[0]], daily_close[:-1]])))
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr_14 = tr.ewm(span=14, adjust=False, min_periods=14).mean().values
    
    # Align HTF indicators to 6h timeframe with proper delay
    williams_r_6h = align_htf_to_ltf(prices, df_1d, williams_r)
    atr_14_6h = align_htf_to_ltf(prices, df_1d, atr_14)
    
    # Calculate 6h RSI(7) for momentum confirmation
    delta = pd.Series(close).diff()
    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)
    avg_gain = gain.ewm(alpha=1/7, adjust=False, min_periods=7).mean()
    avg_loss = loss.ewm(alpha=1/7, adjust=False, min_periods=7).mean()
    rs = avg_gain / (avg_loss + 1e-10)
    rsi_7 = 100 - (100 / (1 + rs))
    rsi_7_values = rsi_7.values
    
    # Calculate 6h volume ratio (current vs 20-period average)
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_ratio = volume / (vol_ma_20 + 1e-10)
    
    signals = np.zeros(n)
    
    for i in range(100, n):
        # Skip if any required data is NaN
        if (np.isnan(williams_r_6h[i]) or np.isnan(atr_14_6h[i]) or 
            np.isnan(rsi_7_values[i]) or np.isnan(volume_ratio[i])):
            signals[i] = 0.0
            continue
        
        # Entry conditions:
        # 1. Mean reversion: Williams %R shows extreme oversold/overbought
        # 2. Momentum confirmation: RSI not in extreme territory (avoid catching falling knife)
        # 3. Volatility filter: ATR > 0.5% of price (avoid low volatility chop)
        # 4. Volume confirmation: volume > 1.5x average
        # 5. Discrete position sizing: 0.25
        
        # Long conditions: Williams %R oversold (< -80) + RSI recovering (> 30)
        if (williams_r_6h[i] < -80 and            # Oversold condition
            rsi_7_values[i] > 30 and              # RSI not in extreme oversold
            volume_ratio[i] > 1.5 and             # Volume confirmation
            atr_14_6h[i] > 0.005 * close[i]):     # Volatility filter
            signals[i] = 0.25
            
        # Short conditions: Williams %R overbought (> -20) + RSI declining (< 70)
        elif (williams_r_6h[i] > -20 and          # Overbought condition
              rsi_7_values[i] < 70 and            # RSI not in extreme overbought
              volume_ratio[i] > 1.5 and           # Volume confirmation
              atr_14_6h[i] > 0.005 * close[i]):   # Volatility filter
            signals[i] = -0.25
        else:
            signals[i] = 0.0
    
    return signals

name = "6h_WilliamsR_MeanReversion_RSI_Filter"
timeframe = "6h"
leverage = 1.0
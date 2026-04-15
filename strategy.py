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
    
    # Calculate daily RSI(14) for regime filter
    delta = pd.Series(daily_close).diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(alpha=1/14, adjust=False, min_periods=14).mean()
    avg_loss = loss.ewm(alpha=1/14, adjust=False, min_periods=14).mean()
    rs = avg_gain / (avg_loss + 1e-10)
    rsi_14 = 100 - (100 / (1 + rs))
    rsi_14_values = rsi_14.values
    
    # Align HTF indicators to 6h timeframe with proper delay
    atr_14_6h = align_htf_to_ltf(prices, df_1d, atr_14)
    rsi_14_6h = align_htf_to_ltf(prices, df_1d, rsi_14_values)
    
    # Calculate 6h Donchian channels (20-period) for breakout signals
    highest_20 = pd.Series(high).rolling(window=20, min_periods=20).max().values
    lowest_20 = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # Calculate 6h volume ratio (current vs 20-period average)
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_ratio = volume / (vol_ma_20 + 1e-10)
    
    signals = np.zeros(n)
    
    for i in range(100, n):
        # Skip if any required data is NaN
        if (np.isnan(atr_14_6h[i]) or np.isnan(rsi_14_6h[i]) or 
            np.isnan(highest_20[i]) or np.isnan(lowest_20[i]) or np.isnan(volume_ratio[i])):
            signals[i] = 0.0
            continue
        
        # Entry conditions:
        # Regime filter: Only trade when RSI < 40 (oversold) or RSI > 60 (overbought)
        # Long: 6h price breaks above Donchian high with volume confirmation in oversold regime
        # Short: 6h price breaks below Donchian low with volume confirmation in overbought regime
        # Volatility filter: ATR > 0.5% of price (avoid low volatility chop)
        # Volume confirmation: volume > 1.5x average
        # Discrete position sizing: 0.25
        
        # Long conditions: 6h breakout above Donchian high in oversold regime
        if (close[i] > highest_20[i] and            # 6h price above Donchian high
            rsi_14_6h[i] < 40 and                   # Oversold regime (daily RSI < 40)
            volume_ratio[i] > 1.5 and               # Volume confirmation
            atr_14_6h[i] > 0.005 * close[i]):       # Volatility filter
            signals[i] = 0.25
            
        # Short conditions: 6h breakdown below Donchian low in overbought regime
        elif (close[i] < lowest_20[i] and           # 6h price below Donchian low
              rsi_14_6h[i] > 60 and                 # Overbought regime (daily RSI > 60)
              volume_ratio[i] > 1.5 and             # Volume confirmation
              atr_14_6h[i] > 0.005 * close[i]):     # Volatility filter
            signals[i] = -0.25
        else:
            signals[i] = 0.0
    
    return signals

name = "6h_Donchian_Breakout_RSI_Regime_Volume_Filter"
timeframe = "6h"
leverage = 1.0
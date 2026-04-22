#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Load 1d data for ATR-based volatility regime (ONCE before loop)
    df_1d = get_htf_data(prices, '1d')
    
    if len(df_1d) < 20:
        return np.zeros(n)
    
    # Calculate 14-period ATR on daily timeframe for volatility regime
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    tr1 = high_1d - low_1d
    tr2 = np.abs(high_1d - np.roll(close_1d, 1))
    tr3 = np.abs(low_1d - np.roll(close_1d, 1))
    tr2[0] = tr1[0]
    tr3[0] = tr1[0]
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    atr_14 = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    # Calculate 50-period SMA of ATR for regime threshold
    atr_ma50 = pd.Series(atr_14).rolling(window=50, min_periods=50).mean().values
    
    # Align ATR regime to 12h timeframe
    atr_14_aligned = align_htf_to_ltf(prices, df_1d, atr_14)
    atr_ma50_aligned = align_htf_to_ltf(prices, df_1d, atr_ma50)
    
    # Volatility regime: high volatility when ATR > 1.2 * ATR_MA50
    vol_regime = atr_14_aligned > (1.2 * atr_ma50_aligned)
    
    # 12-period RSI for mean reversion signals
    delta = np.diff(close, prepend=close[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    avg_gain = pd.Series(gain).rolling(window=12, min_periods=12).mean().values
    avg_loss = pd.Series(loss).rolling(window=12, min_periods=12).mean().values
    rs = np.where(avg_loss != 0, avg_gain / avg_loss, 0)
    rsi = 100 - (100 / (1 + rs))
    
    # Volume confirmation: 20-period average
    vol_avg_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(1, n):
        # Skip if data not ready
        if (np.isnan(rsi[i]) or np.isnan(vol_avg_20[i]) or 
            np.isnan(atr_14_aligned[i]) or np.isnan(atr_ma50_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: RSI oversold (< 30) + high volatility regime + volume spike
            if (rsi[i] < 30 and 
                vol_regime[i] and
                volume[i] > 1.5 * vol_avg_20[i]):
                signals[i] = 0.25
                position = 1
            # Short: RSI overbought (> 70) + high volatility regime + volume spike
            elif (rsi[i] > 70 and 
                  vol_regime[i] and
                  volume[i] > 1.5 * vol_avg_20[i]):
                signals[i] = -0.25
                position = -1
        else:
            # Exit: RSI returns to neutral zone (40-60) or volatility regime ends
            if position == 1:
                # Exit long: RSI crosses above 40 or volatility regime ends
                if rsi[i] > 40 or not vol_regime[i]:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = 0.25
            else:  # position == -1
                # Exit short: RSI crosses below 60 or volatility regime ends
                if rsi[i] < 60 or not vol_regime[i]:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = -0.25
    
    return signals

name = "12H_RSI_VolRegime_Volume_MeanReversion"
timeframe = "12h"
leverage = 1.0
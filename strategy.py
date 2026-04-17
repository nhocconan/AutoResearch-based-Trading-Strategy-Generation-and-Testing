#!/usr/bin/env python3
"""
4h_RSI20_MeanReversion_VolumeSpike_V1
Mean reversion on 4h using RSI(20) + volume spike + volatility filter.
Long when RSI<30 + volume>1.5x MA + ATR<mean ATR (low volatility).
Short when RSI>70 + volume>1.5x MA + ATR<mean ATR.
Exit when RSI crosses 50.
Position size: 0.25. Target: 20-40 trades/year.
Works in bull/bear: mean reversion works in ranging markets, volatility filter avoids false signals in high volatility.
"""

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
    
    # RSI(20) calculation
    delta = np.diff(close, prepend=close[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    avg_gain = pd.Series(gain).ewm(alpha=1/20, adjust=False, min_periods=20).mean().values
    avg_loss = pd.Series(loss).ewm(alpha=1/20, adjust=False, min_periods=20).mean().values
    rs = avg_gain / (avg_loss + 1e-10)
    rsi = 100 - (100 / (1 + rs))
    
    # 1d ATR for volatility filter
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    tr1 = high_1d - low_1d
    tr2 = np.abs(high_1d - np.roll(close_1d, 1))
    tr3 = np.abs(low_1d - np.roll(close_1d, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]
    atr_1d = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    atr_mean_1d = pd.Series(atr_1d).rolling(window=20, min_periods=20).mean().values
    atr_mean_1d_aligned = align_htf_to_ltf(prices, df_1d, atr_mean_1d)
    
    # 1d volume filter
    volume_1d = df_1d['volume'].values
    volume_ma20_1d = pd.Series(volume_1d).rolling(window=20, min_periods=20).mean().values
    volume_ma20_1d_aligned = align_htf_to_ltf(prices, df_1d, volume_ma20_1d)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(40, n):  # warmup for RSI and rolling
        # Skip if any required data is not available
        if np.isnan(rsi[i]) or np.isnan(atr_mean_1d_aligned[i]) or np.isnan(volume_ma20_1d_aligned[i]):
            signals[i] = 0.0
            continue
        
        # Get current 1d ATR and volume (aligned to 4h)
        atr_1d_current = align_htf_to_ltf(prices, df_1d, atr_1d)[i]
        vol_1d_current = align_htf_to_ltf(prices, df_1d, volume_1d)[i]
        
        volatility_filter = atr_1d_current < atr_mean_1d_aligned[i]  # low volatility
        volume_filter = vol_1d_current > (1.5 * volume_ma20_1d_aligned[i])
        
        if position == 0:
            # Long when RSI<30 (oversold) + volume spike + low volatility
            if rsi[i] < 30 and volume_filter and volatility_filter:
                signals[i] = 0.25
                position = 1
            # Short when RSI>70 (overbought) + volume spike + low volatility
            elif rsi[i] > 70 and volume_filter and volatility_filter:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit long: RSI crosses above 50
            if rsi[i] > 50:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: RSI crosses below 50
            if rsi[i] < 50:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "4h_RSI20_MeanReversion_VolumeSpike_V1"
timeframe = "4h"
leverage = 1.0
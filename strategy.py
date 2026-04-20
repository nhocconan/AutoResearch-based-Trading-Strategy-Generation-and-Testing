#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Load daily data for HTF analysis
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    volume_1d = df_1d['volume'].values
    
    # 14-period RSI on daily closes (momentum/mean reversion hybrid)
    delta = pd.Series(close_1d).diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(alpha=1/14, adjust=False, min_periods=14).mean()
    avg_loss = loss.ewm(alpha=1/14, adjust=False, min_periods=14).mean()
    rs = avg_gain / avg_loss
    rsi_1d = 100 - (100 / (1 + rs))
    rsi_1d = rsi_1d.fillna(50).values  # Neutral when undefined
    rsi_1d_aligned = align_htf_to_ltf(prices, df_1d, rsi_1d)
    
    # Daily ATR(14) for volatility filter
    tr1 = high_1d - low_1d
    tr2 = np.abs(high_1d - np.roll(close_1d, 1))
    tr3 = np.abs(low_1d - np.roll(close_1d, 1))
    tr2[0] = tr1[0]
    tr3[0] = tr1[0]
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    atr_14 = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    atr_14_aligned = align_htf_to_ltf(prices, df_1d, atr_14)
    
    # Daily volume ratio (current vs 20-day average)
    vol_ma_20 = pd.Series(volume_1d).rolling(window=20, min_periods=20).mean().values
    vol_ratio = volume_1d / np.where(vol_ma_20 == 0, 1, vol_ma_20)
    vol_ratio_aligned = align_htf_to_ltf(prices, df_1d, vol_ratio)
    
    # 6h price data (primary timeframe)
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(100, n):
        # Skip if NaN in critical values
        if (np.isnan(rsi_1d_aligned[i]) or np.isnan(atr_14_aligned[i]) or 
            np.isnan(vol_ratio_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        price = close[i]
        vol = volume[i]
        
        rsi = rsi_1d_aligned[i]
        atr = atr_14_aligned[i]
        vol_ratio = vol_ratio_aligned[i]
        
        # Volatility filter: trade only when ATR is in normal range
        atr_ma_20 = pd.Series(atr_14_aligned).rolling(window=20, min_periods=20).mean().values[i]
        vol_filter = (atr > 0.3 * atr_ma_20) and (atr < 3.0 * atr_ma_20)
        
        # Volume filter: require above-average volume
        vol_filter = vol_filter and (vol_ratio > 1.3)
        
        if position == 0:
            # Enter long when RSI shows bullish momentum with volume confirmation
            if (rsi > 55) and vol_filter:
                signals[i] = 0.25
                position = 1
            # Enter short when RSI shows bearish momentum with volume confirmation
            elif (rsi < 45) and vol_filter:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit long when RSI turns bearish or volatility spikes
            if (rsi < 40) or (atr > 4.0 * atr_ma_20):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short when RSI turns bullish or volatility spikes
            if (rsi > 60) or (atr > 4.0 * atr_ma_20):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "6h_RSI_Momentum_VolumeFilter_v1"
timeframe = "6h"
leverage = 1.0
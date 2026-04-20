#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Load daily data once for 1d indicators
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    volume_1d = df_1d['volume'].values
    
    # Calculate ATR(14) on daily
    tr1 = np.abs(high_1d - low_1d)
    tr2 = np.abs(high_1d - np.roll(close_1d, 1))
    tr3 = np.abs(low_1d - np.roll(close_1d, 1))
    tr1[0] = high_1d[0] - low_1d[0]
    tr2[0] = np.abs(high_1d[0] - close_1d[0])
    tr3[0] = np.abs(low_1d[0] - close_1d[0])
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    atr_1d = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    atr_1d_aligned = align_htf_to_ltf(prices, df_1d, atr_1d)
    
    # Calculate 20-period SMA of daily volume
    vol_sma_1d = pd.Series(volume_1d).rolling(window=20, min_periods=20).mean().values
    vol_sma_1d_aligned = align_htf_to_ltf(prices, df_1d, vol_sma_1d)
    
    # Calculate 50-period SMA of daily close (trend filter)
    sma50_1d = pd.Series(close_1d).rolling(window=50, min_periods=50).mean().values
    sma50_1d_aligned = align_htf_to_ltf(prices, df_1d, sma50_1d)
    
    # Calculate 12-period RSI on daily close
    delta = np.diff(close_1d, prepend=close_1d[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    avg_gain = pd.Series(gain).rolling(window=14, min_periods=14).mean().values
    avg_loss = pd.Series(loss).rolling(window=14, min_periods=14).mean().values
    rs = np.where(avg_loss != 0, avg_gain / avg_loss, 0)
    rsi_1d = 100 - (100 / (1 + rs))
    rsi_1d_aligned = align_htf_to_ltf(prices, df_1d, rsi_1d)
    
    # Main timeframe data (12h)
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(50, n):
        # Skip if NaN in critical values
        if (np.isnan(atr_1d_aligned[i]) or np.isnan(vol_sma_1d_aligned[i]) or 
            np.isnan(sma50_1d_aligned[i]) or np.isnan(rsi_1d_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        price = close[i]
        atr_1d = atr_1d_aligned[i]
        vol_sma_1d = vol_sma_1d_aligned[i]
        sma50_1d = sma50_1d_aligned[i]
        rsi_1d = rsi_1d_aligned[i]
        vol_current = volume[i]
        
        # Range filter: ATR below 30-day average (use 30-period SMA of ATR)
        atr_ma_30 = pd.Series(atr_1d).rolling(window=30, min_periods=30).mean().values
        atr_ma_30_aligned = align_htf_to_ltf(prices, df_1d, atr_ma_30)
        atr_ma_30_val = atr_ma_30_aligned[i] if not np.isnan(atr_ma_30_aligned[i]) else atr_1d
        range_filter = atr_1d < atr_ma_30_val  # Low volatility range
        
        # Volume filter: current volume > 1.3x daily average
        vol_filter = vol_current > 1.3 * vol_sma_1d
        
        # Trend filter: price above/below 50-day SMA
        uptrend = price > sma50_1d
        downtrend = price < sma50_1d
        
        # RSI filters: avoid extremes
        rsi_not_overbought = rsi_1d < 70
        rsi_not_oversold = rsi_1d > 30
        
        if position == 0:
            # Long: in uptrend, not overbought, volume spike, in range
            if uptrend and rsi_not_overbought and vol_filter and range_filter:
                signals[i] = 0.25
                position = 1
            # Short: in downtrend, not oversold, volume spike, in range
            elif downtrend and rsi_not_oversold and vol_filter and range_filter:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Long exit: trend breaks OR RSI overbought OR volatility breaks
            if not uptrend or rsi_1d >= 70 or not range_filter:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: trend breaks OR RSI oversold OR volatility breaks
            if not downtrend or rsi_1d <= 30 or not range_filter:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "12h_1d_ATR_Range_Volume_RSI_Filter_v1"
timeframe = "12h"
leverage = 1.0
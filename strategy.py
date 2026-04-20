#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Load weekly data ONCE
    df_1w = get_htf_data(prices, '1w')
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    
    # Weekly SMA(20) - used as support/resistance
    close_1w_series = pd.Series(close_1w)
    sma_20_1w = close_1w_series.rolling(window=20, min_periods=20).mean().values
    
    # Load daily data ONCE
    df_1d = get_htf_data(prices, '1d')
    close_1d = df_1d['close'].values
    volume_1d = df_1d['volume'].values
    
    # Daily RSI(14) - momentum
    delta = np.diff(close_1d)
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    avg_gain = np.concatenate([[np.nan], np.convolve(gain, np.ones(14)/14, mode='full')[:len(gain)]])
    avg_loss = np.concatenate([[np.nan], np.convolve(loss, np.ones(14)/14, mode='full')[:len(loss)]])
    rs = np.where(avg_loss != 0, avg_gain / avg_loss, 0)
    rsi_14_1d = 100 - (100 / (1 + rs))
    # First 14 values invalid
    rsi_14_1d[:14] = np.nan
    
    # Align weekly and daily indicators to daily timeframe
    sma_20_1w_aligned = align_htf_to_ltf(df_1d, df_1w, sma_20_1w)
    rsi_14_1d_aligned = align_htf_to_ltf(df_1d, df_1d, rsi_14_1d)
    
    # Daily price and volume
    close = df_1d
    volume = volume_1d
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(100, n):
        # Skip if NaN
        if np.isnan(sma_20_1w_aligned[i]) or np.isnan(rsi_14_1d_aligned[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        sma_20_val = sma_20_1w_aligned[i]
        rsi_val = rsi_14_1d_aligned[i]
        price = close[i]
        vol = volume[i]
        
        # Volume filter: current volume must be above 20-day average
        if i >= 20:
            vol_ma_20 = np.mean(volume[i-20:i])
            vol_filter = vol > vol_ma_20
        else:
            vol_filter = False
        
        if position == 0:
            # Long: price above weekly SMA20, RSI not overbought, volume confirmation
            if price > sma_20_val and rsi_val < 70 and vol_filter:
                signals[i] = 0.25
                position = 1
            # Short: price below weekly SMA20, RSI not oversold, volume confirmation
            elif price < sma_20_val and rsi_val > 30 and vol_filter:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Long exit: price below weekly SMA20 or RSI overbought
            if price < sma_20_val or rsi_val > 70:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: price above weekly SMA20 or RSI oversold
            if price > sma_20_val or rsi_val < 30:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "1d_WeeklySMA20_RSI14_VolumeFilter"
timeframe = "1d"
leverage = 1.0
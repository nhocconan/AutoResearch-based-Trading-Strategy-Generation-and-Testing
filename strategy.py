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
    
    # Get daily data for calculations
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate daily volatility (ATR-like)
    tr1 = high_1d - low_1d
    tr2 = np.abs(high_1d - np.roll(close_1d, 1))
    tr3 = np.abs(low_1d - np.roll(close_1d, 1))
    tr1[0] = np.nan
    tr2[0] = np.nan
    tr3[0] = np.nan
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    atr_1d = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    # Calculate daily EMA of close
    ema_1d = pd.Series(close_1d).ewm(span=21, adjust=False, min_periods=21).mean().values
    
    # Align daily EMA to 1d timeframe (no additional delay needed for EMA)
    ema_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_1d)
    
    # Calculate price position relative to EMA (normalized by ATR)
    # Positive = price above EMA, Negative = price below EMA
    price_vs_ema = (close_1d - ema_1d) / atr_1d
    price_vs_ema[0:21] = np.nan  # Not enough data for EMA
    
    # Align the normalized price position to 1d
    price_vs_ema_aligned = align_htf_to_ltf(prices, df_1d, price_vs_ema)
    
    # Calculate 10-day RSI on daily close
    delta = pd.Series(close_1d).diff()
    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)
    avg_gain = gain.rolling(window=10, min_periods=10).mean()
    avg_loss = loss.rolling(window=10, min_periods=10).mean()
    rs = avg_gain / avg_loss
    rsi_1d = 100 - (100 / (1 + rs))
    rsi_1d = rsi_1d.values
    rsi_1d[0:10] = np.nan
    
    # Align RSI to 1d
    rsi_1d_aligned = align_htf_to_ltf(prices, df_1d, rsi_1d)
    
    # Volume confirmation: today's volume > 1.5x 20-day average
    volume_ma20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # -1: short, 0: flat, 1: long
    
    start_idx = 30  # Need EMA21, RSI10, volume MA20
    
    for i in range(start_idx, n):
        # Skip if any required data is not available
        if (np.isnan(volume_ma20[i]) or 
            np.isnan(ema_1d_aligned[i]) or 
            np.isnan(price_vs_ema_aligned[i]) or
            np.isnan(rsi_1d_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Volume filter: current volume > 1.5x 20-day average
        volume_filter = volume[i] > (1.5 * volume_ma20[i])
        
        if position == 0:
            # Long: price above EMA (uptrend) AND RSI oversold (<30) AND volume confirmation
            if (price_vs_ema_aligned[i] > 0 and 
                rsi_1d_aligned[i] < 30 and 
                volume_filter):
                signals[i] = 0.25
                position = 1
            # Short: price below EMA (downtrend) AND RSI overbought (>70) AND volume confirmation
            elif (price_vs_ema_aligned[i] < 0 and 
                  rsi_1d_aligned[i] > 70 and 
                  volume_filter):
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit long: price crosses below EMA OR RSI overbought (>70)
            if (price_vs_ema_aligned[i] < 0 or 
                rsi_1d_aligned[i] > 70):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: price crosses above EMA OR RSI oversold (<30)
            if (price_vs_ema_aligned[i] > 0 or 
                rsi_1d_aligned[i] < 30):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "1d_EMA_RSI_Volume_Filter"
timeframe = "1d"
leverage = 1.0
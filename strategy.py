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
    
    # Load daily data ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate 14-period RSI on daily close
    delta = pd.Series(df_1d['close']).diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14, min_periods=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14, min_periods=14).mean()
    rs = gain / loss
    rsi_14 = (100 - (100 / (1 + rs))).values
    rsi_14_aligned = align_htf_to_ltf(prices, df_1d, rsi_14)
    
    # Calculate 200-period SMA on daily close for trend filter
    sma_200_1d = pd.Series(df_1d['close']).rolling(window=200, min_periods=200).mean().values
    sma_200_1d_aligned = align_htf_to_ltf(prices, df_1d, sma_200_1d)
    
    # Calculate 14-period ATR for volatility filter
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr2[0] = 0
    tr3[0] = 0
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    atr = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    # Calculate 20-period average volume for confirmation
    vol_avg = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0
    position_size = 0.25  # 25% position size
    
    # Start after enough data for calculations
    start = 200
    
    for i in range(start, n):
        # Skip if any critical data is NaN
        if (np.isnan(rsi_14_aligned[i]) or 
            np.isnan(sma_200_1d_aligned[i]) or 
            np.isnan(atr[i]) or 
            np.isnan(vol_avg[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        vol = volume[i]
        
        # Volatility filter: avoid extremely low volatility periods
        atr_ratio = atr[i] / price if price > 0 else 0
        vol_filter = atr_ratio > 0.003  # Minimum 0.3% ATR relative to price
        
        # Volume confirmation: current volume > 1.5x average
        vol_confirm = vol > (vol_avg[i] * 1.5)
        
        # RSI extreme + SMA filter: RSI < 30 and price > SMA200 for long, RSI > 70 and price < SMA200 for short
        rsi_long = rsi_14_aligned[i] < 30
        rsi_short = rsi_14_aligned[i] > 70
        sma_filter_long = price > sma_200_1d_aligned[i]
        sma_filter_short = price < sma_200_1d_aligned[i]
        
        if position == 0:
            # Long setup: RSI oversold + price above SMA200 + volume confirmation + volatility filter
            if (rsi_long and sma_filter_long and vol_confirm and vol_filter):
                position = 1
                signals[i] = position_size
            # Short setup: RSI overbought + price below SMA200 + volume confirmation + volatility filter
            elif (rsi_short and sma_filter_short and vol_confirm and vol_filter):
                position = -1
                signals[i] = -position_size
            else:
                signals[i] = 0.0
        elif position == 1:
            # Exit long: RSI crosses above 50 or price crosses below SMA200
            if (rsi_14_aligned[i] > 50) or (price < sma_200_1d_aligned[i]):
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = position_size
        elif position == -1:
            # Exit short: RSI crosses below 50 or price crosses above SMA200
            if (rsi_14_aligned[i] < 50) or (price > sma_200_1d_aligned[i]):
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -position_size
    
    return signals

name = "1d_RSI_SMA200_Volume_Filter"
timeframe = "1d"
leverage = 1.0
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
    
    # Get 1d data for calculations
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    volume_1d = df_1d['volume'].values
    
    # Calculate 14-period ATR on 1d (volatility filter)
    tr1 = high_1d - low_1d
    tr2 = np.abs(high_1d - np.roll(close_1d, 1))
    tr3 = np.abs(low_1d - np.roll(close_1d, 1))
    tr1[0] = 0
    tr2[0] = 0
    tr3[0] = 0
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    atr_14 = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    # Calculate 50-period SMA on 1d (trend filter)
    sma_50_1d = pd.Series(close_1d).rolling(window=50, min_periods=50).mean().values
    
    # Calculate RSI(14) on 1d
    delta = np.diff(close_1d, prepend=close_1d[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    avg_gain = pd.Series(gain).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    avg_loss = pd.Series(loss).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    rs = avg_gain / (avg_loss + 1e-10)
    rsi_14 = 100 - (100 / (1 + rs))
    
    # Calculate volume ratio: current volume / 20-period average volume
    vol_ma_20 = pd.Series(volume_1d).rolling(window=20, min_periods=20).mean().values
    vol_ratio = volume_1d / (vol_ma_20 + 1e-10)
    
    # Align indicators to 12h timeframe
    atr_14_aligned = align_htf_to_ltf(prices, df_1d, atr_14)
    sma_50_aligned = align_htf_to_ltf(prices, df_1d, sma_50_1d)
    rsi_14_aligned = align_htf_to_ltf(prices, df_1d, rsi_14)
    vol_ratio_aligned = align_htf_to_ltf(prices, df_1d, vol_ratio)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    position_size = 0.25  # 25% of capital
    
    for i in range(100, n):
        # Skip if data not ready
        if (np.isnan(atr_14_aligned[i]) or 
            np.isnan(sma_50_aligned[i]) or
            np.isnan(rsi_14_aligned[i]) or
            np.isnan(vol_ratio_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Trend filter: price above/below SMA50
        above_sma = close[i] > sma_50_aligned[i]
        below_sma = close[i] < sma_50_aligned[i]
        
        # RSI conditions: avoid extremes, favor mean reversion in range
        rsi_not_overbought = rsi_14_aligned[i] < 65
        rsi_not_oversold = rsi_14_aligned[i] > 35
        
        # Volatility filter: require sufficient but not excessive volatility
        vol_ok = atr_14_aligned[i] > 0
        
        # Volume confirmation: require above average volume
        vol_confirm = vol_ratio_aligned[i] > 1.2
        
        # Entry conditions: mean reversion with volume confirmation
        long_entry = below_sma and rsi_not_oversold and vol_ok and vol_confirm
        short_entry = above_sma and rsi_not_overbought and vol_ok and vol_confirm
        
        # Exit conditions: trend reversal or RSI extreme
        exit_long = position == 1 and (above_sma or rsi_14_aligned[i] > 75)
        exit_short = position == -1 and (below_sma or rsi_14_aligned[i] < 25)
        
        # Execute signals
        if long_entry and position != 1:
            position = 1
            signals[i] = position_size
        elif short_entry and position != -1:
            position = -1
            signals[i] = -position_size
        elif exit_long or exit_short:
            position = 0
            signals[i] = 0.0
        # Hold current position
        else:
            if position == 1:
                signals[i] = position_size
            elif position == -1:
                signals[i] = -position_size
            else:
                signals[i] = 0.0
    
    return signals

name = "12h_sma50_rsi14_vol_confirm_v1"
timeframe = "12h"
leverage = 1.0
#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 200:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Load 1d data once
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # Calculate 14-period RSI on 1d
    delta = np.diff(df_1d['close'].values, prepend=df_1d['close'].values[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    avg_gain = pd.Series(gain).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    avg_loss = pd.Series(loss).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    rs = avg_gain / (avg_loss + 1e-10)
    rsi_1d = 100 - (100 / (1 + rs))
    
    # Calculate 20-period standard deviation for Bollinger Bands on 1d
    close_1d = df_1d['close'].values
    sma_20_1d = pd.Series(close_1d).rolling(window=20, min_periods=20).mean().values
    std_20_1d = pd.Series(close_1d).rolling(window=20, min_periods=20).std().values
    upper_bb_1d = sma_20_1d + 2 * std_20_1d
    lower_bb_1d = sma_20_1d - 2 * std_20_1d
    
    # Calculate 4-period volume moving average (4h periods in a day)
    vol_ma_4 = pd.Series(volume).rolling(window=4, min_periods=4).mean().values
    
    signals = np.zeros(n)
    position = 0
    position_size = 0.25  # 25% position size
    
    for i in range(200, n):
        # Get aligned indicators
        rsi_1d_aligned = align_htf_to_ltf(prices, df_1d, rsi_1d)[i]
        upper_bb_1d_aligned = align_htf_to_ltf(prices, df_1d, upper_bb_1d)[i]
        lower_bb_1d_aligned = align_htf_to_ltf(prices, df_1d, lower_bb_1d)[i]
        vol_ma_4_val = vol_ma_4[i]  # already LTF
        
        # Check for NaN values
        if (np.isnan(rsi_1d_aligned) or np.isnan(upper_bb_1d_aligned) or 
            np.isnan(lower_bb_1d_aligned) or np.isnan(vol_ma_4_val)):
            continue
        
        # Volume confirmation (> 1.3x average)
        volume_confirm = volume[i] > 1.3 * vol_ma_4_val
        
        if position == 0:  # No position - look for entries
            if volume_confirm:
                # Long: Pullback to support (price touches lower BB + RSI oversold)
                if close[i] <= lower_bb_1d_aligned and rsi_1d_aligned < 30:
                    position = 1
                    signals[i] = position_size
                # Short: Pullback to resistance (price touches upper BB + RSI overbought)
                elif close[i] >= upper_bb_1d_aligned and rsi_1d_aligned > 70:
                    position = -1
                    signals[i] = -position_size
        elif position == 1:  # Long position - exit when RSI overbought
            if rsi_1d_aligned > 70:
                position = 0
                signals[i] = 0.0
        elif position == -1:  # Short position - exit when RSI oversold
            if rsi_1d_aligned < 30:
                position = 0
                signals[i] = 0.0
    
    return signals

name = "4h_RSI_BB_Pullback_v1"
timeframe = "4h"
leverage = 1.0
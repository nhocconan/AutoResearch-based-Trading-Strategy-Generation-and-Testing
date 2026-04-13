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
    
    # Get 1d data for ATR and Bollinger Bands
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate 1d ATR(14)
    tr_1d = np.maximum(high_1d[1:] - low_1d[1:], 
                       np.maximum(np.abs(high_1d[1:] - close_1d[:-1]), 
                                  np.abs(low_1d[1:] - close_1d[:-1])))
    tr_1d = np.concatenate([[tr_1d[0]], tr_1d])
    atr_1d = np.zeros_like(close_1d)
    atr_1d[0] = tr_1d[0]
    for i in range(1, len(tr_1d)):
        atr_1d[i] = 0.93 * atr_1d[i-1] + 0.07 * tr_1d[i]
    
    # Calculate 1d Bollinger Bands (20, 2)
    sma_20 = pd.Series(close_1d).rolling(window=20, min_periods=20).mean().values
    std_20 = pd.Series(close_1d).rolling(window=20, min_periods=20).std().values
    upper_bb = sma_20 + 2 * std_20
    lower_bb = sma_20 - 2 * std_20
    
    # Calculate 1d RSI(14)
    delta = np.diff(close_1d)
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    avg_gain = pd.Series(gain).rolling(window=14, min_periods=14).mean().values
    avg_loss = pd.Series(loss).rolling(window=14, min_periods=14).mean().values
    rs = np.where(avg_loss != 0, avg_gain / avg_loss, 0)
    rsi_1d = 100 - (100 / (1 + rs))
    # Pad for the first element
    rsi_1d = np.concatenate([[50], rsi_1d])
    
    # Align all indicators to 6h timeframe
    atr_1d_aligned = align_htf_to_ltf(prices, df_1d, atr_1d)
    upper_bb_aligned = align_htf_to_ltf(prices, df_1d, upper_bb)
    lower_bb_aligned = align_htf_to_ltf(prices, df_1d, lower_bb)
    rsi_1d_aligned = align_htf_to_ltf(prices, df_1d, rsi_1d)
    
    signals = np.zeros(n)
    position = 0
    position_size = 0.25
    entry_price = np.full(n, np.nan)
    
    for i in range(30, n):
        # Skip if data not ready
        if (np.isnan(atr_1d_aligned[i]) or 
            np.isnan(upper_bb_aligned[i]) or
            np.isnan(lower_bb_aligned[i]) or
            np.isnan(rsi_1d_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Entry conditions: Bollinger Band squeeze + RSI extreme
        bb_width = upper_bb_aligned[i] - lower_bb_aligned[i]
        bb_width_percentile = pd.Series(bb_width[:i+1]).rolling(window=50, min_periods=10).rank(pct=True).iloc[-1] if i >= 10 else 0.5
        
        # Volatility squeeze condition (BB width in lower 20th percentile)
        volatility_squeeze = bb_width_percentile < 0.2
        
        # RSI extreme conditions
        rsi_oversold = rsi_1d_aligned[i] < 30
        rsi_overbought = rsi_1d_aligned[i] > 70
        
        # Entry signals
        long_entry = volatility_squeeze and rsi_oversold and close[i] > close[i-1]
        short_entry = volatility_squeeze and rsi_overbought and close[i] < close[i-1]
        
        # Exit conditions: RSI mean reversion or volatility expansion
        exit_long = position == 1 and (rsi_1d_aligned[i] > 50 or bb_width_percentile > 0.8)
        exit_short = position == -1 and (rsi_1d_aligned[i] < 50 or bb_width_percentile > 0.8)
        
        # Execute signals
        if long_entry and position != 1:
            position = 1
            signals[i] = position_size
            entry_price[i] = close[i]
        elif short_entry and position != -1:
            position = -1
            signals[i] = -position_size
            entry_price[i] = close[i]
        elif position == 1 and exit_long:
            position = 0
            signals[i] = 0.0
            entry_price[i] = np.nan
        elif position == -1 and exit_short:
            position = 0
            signals[i] = 0.0
            entry_price[i] = np.nan
        # Hold current position
        else:
            if position == 1:
                signals[i] = position_size
                entry_price[i] = entry_price[i-1] if i > 0 else np.nan
            elif position == -1:
                signals[i] = -position_size
                entry_price[i] = entry_price[i-1] if i > 0 else np.nan
            else:
                signals[i] = 0.0
                entry_price[i] = np.nan
    
    return signals

name = "6h_1d_bb_rsi_squeeze_v1"
timeframe = "6h"
leverage = 1.0
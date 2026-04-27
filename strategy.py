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
    
    # Get weekly data for higher timeframe context (1w)
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 50:
        return np.zeros(n)
    
    close_1w = df_1w['close'].values
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    
    # Calculate weekly ATR(10) for volatility
    tr1 = high_1w - low_1w
    tr2 = np.abs(high_1w - np.roll(close_1w, 1))
    tr3 = np.abs(low_1w - np.roll(close_1w, 1))
    tr2[0] = tr1[0]
    tr3[0] = tr1[0]
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    atr_10_1w = pd.Series(tr).rolling(window=10, min_periods=10).mean().values
    atr_10_1w_aligned = align_htf_to_ltf(prices, df_1w, atr_10_1w)
    
    # Calculate weekly Donchian channels (20 periods)
    high_20 = pd.Series(high_1w).rolling(window=20, min_periods=20).max().values
    low_20 = pd.Series(low_1w).rolling(window=20, min_periods=20).min().values
    high_20_aligned = align_htf_to_ltf(prices, df_1w, high_20)
    low_20_aligned = align_htf_to_ltf(prices, df_1w, low_20)
    
    # Calculate weekly RSI(14) for momentum
    delta = pd.Series(close_1w).diff()
    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)
    avg_gain = gain.rolling(window=14, min_periods=14).mean()
    avg_loss = loss.rolling(window=14, min_periods=14).mean()
    rs = avg_gain / avg_loss
    rsi_14_1w = (100 - (100 / (1 + rs))).values
    rsi_14_1w_aligned = align_htf_to_ltf(prices, df_1w, rsi_14_1w)
    
    # Precompute session filter (08-20 UTC)
    hours = prices.index.hour
    session_mask = (hours >= 8) & (hours <= 20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after warmup period
    start_idx = 100
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(atr_10_1w_aligned[i]) or 
            np.isnan(high_20_aligned[i]) or 
            np.isnan(low_20_aligned[i]) or 
            np.isnan(rsi_14_1w_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Session filter: only trade during active hours
        if not session_mask[i]:
            signals[i] = 0.0
            continue
        
        # Volatility filter: avoid low volatility periods
        atr_median = np.nanmedian(atr_10_1w_aligned[:i+1]) if i >= 50 else atr_10_1w_aligned[i]
        sufficient_volatility = atr_10_1w_aligned[i] > atr_median * 0.5
        
        # Breakout conditions
        breakout_long = close[i] > high_20_aligned[i]
        breakout_short = close[i] < low_20_aligned[i]
        
        # Momentum filter: RSI not extreme
        rsi_not_overbought = rsi_14_1w_aligned[i] < 70
        rsi_not_oversold = rsi_14_1w_aligned[i] > 30
        
        # Long conditions: breakout above weekly Donchian high + sufficient volatility + RSI not overbought
        long_condition = (breakout_long and 
                         sufficient_volatility and 
                         rsi_not_overbought)
        
        # Short conditions: breakout below weekly Donchian low + sufficient volatility + RSI not oversold
        short_condition = (breakout_short and 
                          sufficient_volatility and 
                          rsi_not_oversold)
        
        if long_condition and position <= 0:
            signals[i] = 0.25
            position = 1
        elif short_condition and position >= 0:
            signals[i] = -0.25
            position = -1
        # Exit conditions: opposite breakout or volatility collapse
        elif position == 1 and (close[i] < low_20_aligned[i] or not sufficient_volatility):
            signals[i] = 0.0
            position = 0
        elif position == -1 and (close[i] > high_20_aligned[i] or not sufficient_volatility):
            signals[i] = 0.0
            position = 0
        # Hold position
        else:
            if position == 1:
                signals[i] = 0.25
            elif position == -1:
                signals[i] = -0.25
            else:
                signals[i] = 0.0
    
    return signals

name = "6h_WeeklyDonchian20_Breakout_RSIFilter_Vol"
timeframe = "6h"
leverage = 1.0
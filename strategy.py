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
    
    # Get daily data for 200 EMA (HTF)
    df_1d = get_htf_data(prices, '1d')
    close_1d = df_1d['close'].values
    
    # Calculate daily 200 EMA
    ema_200 = pd.Series(close_1d).ewm(span=200, adjust=False, min_periods=200).mean().values
    
    # Align daily 200 EMA to 4h
    ema_200_aligned = align_htf_to_ltf(prices, df_1d, ema_200)
    
    # Get weekly data for ATR (HTF)
    df_1w = get_htf_data(prices, '1w')
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    
    # Calculate weekly ATR (14-period)
    tr1 = high_1w - low_1w
    tr2 = np.abs(high_1w - np.roll(close_1w, 1))
    tr3 = np.abs(low_1w - np.roll(close_1w, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]  # First value
    
    atr_1w = pd.Series(tr).ewm(span=14, adjust=False, min_periods=14).mean().values
    
    # Align weekly ATR to 4h
    atr_1w_aligned = align_htf_to_ltf(prices, df_1w, atr_1w)
    
    # Daily ATR for volatility filter
    tr1_d = df_1d['high'].values - df_1d['low'].values
    tr2_d = np.abs(df_1d['high'].values - np.roll(close_1d, 1))
    tr3_d = np.abs(df_1d['low'].values - np.roll(close_1d, 1))
    tr_d = np.maximum(tr1_d, np.maximum(tr2_d, tr3_d))
    tr_d[0] = tr1_d[0]
    atr_1d = pd.Series(tr_d).ewm(span=14, adjust=False, min_periods=14).mean().values
    atr_1d_aligned = align_htf_to_ltf(prices, df_1d, atr_1d)
    
    # Bollinger Bands on 4h close (20, 2)
    sma_20 = pd.Series(close).rolling(window=20, min_periods=20).mean().values
    std_20 = pd.Series(close).rolling(window=20, min_periods=20).std().values
    upper_bb = sma_20 + 2 * std_20
    lower_bb = sma_20 - 2 * std_20
    
    # Bollinger Band Width for regime detection
    bb_width = (upper_bb - lower_bb) / sma_20
    bb_width_ma50 = pd.Series(bb_width).rolling(window=50, min_periods=50).mean().values
    
    signals = np.zeros(n)
    position = 0  # -1: short, 0: flat, 1: long
    
    start_idx = 200  # Need 200 EMA, Bollinger Bands, ATR
    
    for i in range(start_idx, n):
        # Skip if any required data is not available
        if (np.isnan(ema_200_aligned[i]) or 
            np.isnan(atr_1w_aligned[i]) or 
            np.isnan(atr_1d_aligned[i]) or 
            np.isnan(sma_20[i]) or 
            np.isnan(std_20[i]) or 
            np.isnan(bb_width_ma50[i])):
            signals[i] = 0.0
            continue
        
        # Regime filter: Bollinger Band Width > 50-day average (volatile/trending market)
        volatile_regime = bb_width[i] > bb_width_ma50[i]
        
        # Volatility filter: current volatility not too extreme
        vol_filter = atr_1d_aligned[i] < (3.0 * atr_1w_aligned[i])
        
        if position == 0:
            # Long: price touches lower BB in uptrend (price > 200 EMA) with volatile regime
            if (close[i] <= lower_bb[i] and close[i] > ema_200_aligned[i] and 
                volatile_regime and vol_filter):
                signals[i] = 0.25
                position = 1
            # Short: price touches upper BB in downtrend (price < 200 EMA) with volatile regime
            elif (close[i] >= upper_bb[i] and close[i] < ema_200_aligned[i] and 
                  volatile_regime and vol_filter):
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit long: price returns to SMA(20) or hits trailing stop
            if close[i] >= sma_20[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: price returns to SMA(20) or hits trailing stop
            if close[i] <= sma_20[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "4h_BollingerMeanReversion_VolatileRegime"
timeframe = "4h"
leverage = 1.0
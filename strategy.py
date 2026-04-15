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
    
    # Get 1d HTF data once before loop
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # Calculate 1d EMA200 for trend filter
    close_1d = df_1d['close'].values
    ema200_1d = pd.Series(close_1d).ewm(span=200, adjust=False, min_periods=200).mean().values
    ema200_6h = align_htf_to_ltf(prices, df_1d, ema200_1d)
    
    # Get 1w HTF data for weekly structure
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 30:
        return np.zeros(n)
    
    # Calculate weekly higher highs/lows for trend structure
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    
    # Weekly HH: current high > previous high
    weekly_hh = high_1w > np.concatenate([[high_1w[0]], high_1w[:-1]])
    # Weekly LL: current low < previous low
    weekly_ll = low_1w < np.concatenate([[low_1w[0]], low_1w[:-1]])
    
    # Align weekly structure to 6h
    weekly_hh_6h = align_htf_to_ltf(prices, df_1w, weekly_hh.astype(float))
    weekly_ll_6h = align_htf_to_ltf(prices, df_1w, weekly_ll.astype(float))
    
    # Calculate 6h ATR(14) for volatility
    tr1 = high - low
    tr2 = np.abs(high - np.concatenate([[close[0]], close[:-1]]))
    tr3 = np.abs(low - np.concatenate([[close[0]], close[:-1]]))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    atr_14 = pd.Series(tr).ewm(span=14, adjust=False, min_periods=14).mean().values
    
    # Calculate 6h volume ratio
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_ratio = volume / (vol_ma_20 + 1e-10)
    
    signals = np.zeros(n)
    
    # Session filter: avoid low liquidity periods (22-06 UTC)
    hours = prices.index.hour
    in_session = (hours >= 6) & (hours <= 22)  # 6h-22h UTC
    
    for i in range(100, n):
        # Skip if any required data is NaN
        if (np.isnan(ema200_6h[i]) or np.isnan(weekly_hh_6h[i]) or 
            np.isnan(weekly_ll_6h[i]) or np.isnan(atr_14[i]) or 
            np.isnan(volume_ratio[i]) or not in_session[i]):
            signals[i] = 0.0
            continue
        
        # Determine weekly trend structure
        # Bullish structure: making higher highs
        # Bearish structure: making lower lows
        # Neutral: mixed or unclear
        bullish_structure = weekly_hh_6h[i] > 0.5
        bearish_structure = weekly_ll_6h[i] > 0.5
        
        # Long conditions:
        # 1. Price above 1d EMA200 (bullish bias)
        # 2. Weekly structure shows higher highs (bullish momentum)
        # 3. Volume confirmation: volume > 1.5x average
        # 4. Volatility filter: ATR > 0.3% of price (avoid extremely low volatility)
        if (close[i] > ema200_6h[i] and
            bullish_structure and
            volume_ratio[i] > 1.5 and
            atr_14[i] > 0.003 * close[i]):
            signals[i] = 0.25
            
        # Short conditions:
        # 1. Price below 1d EMA200 (bearish bias)
        # 2. Weekly structure shows lower lows (bearish momentum)
        # 3. Volume confirmation: volume > 1.5x average
        # 4. Volatility filter: ATR > 0.3% of price
        elif (close[i] < ema200_6h[i] and
              bearish_structure and
              volume_ratio[i] > 1.5 and
              atr_14[i] > 0.003 * close[i]):
            signals[i] = -0.25
        else:
            signals[i] = 0.0
    
    return signals

name = "6h_1d_EMA200_1w_Structure_Volume_Filter_v1"
timeframe = "6h"
leverage = 1.0
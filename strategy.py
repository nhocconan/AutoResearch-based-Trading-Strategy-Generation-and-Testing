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
    ema200_1d_aligned = align_htf_to_ltf(prices, df_1d, ema200_1d)
    
    # Calculate 1d RSI(14) for momentum filter
    delta = pd.Series(close_1d).diff()
    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)
    avg_gain = gain.ewm(alpha=1/14, adjust=False, min_periods=14).mean()
    avg_loss = loss.ewm(alpha=1/14, adjust=False, min_periods=14).mean()
    rs = avg_gain / (avg_loss + 1e-10)
    rsi_14_1d = (100 - (100 / (1 + rs))).values
    rsi_14_1d_aligned = align_htf_to_ltf(prices, df_1d, rsi_14_1d)
    
    # Get weekly HTF data for structure
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 20:
        return np.zeros(n)
    
    # Calculate weekly Donchian(20) for breakout signals
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    upper_20_1w = pd.Series(high_1w).rolling(window=20, min_periods=20).max().values
    lower_20_1w = pd.Series(low_1w).rolling(window=20, min_periods=20).min().values
    
    # Align weekly Donchian to 6h
    upper_20_6h = align_htf_to_ltf(prices, df_1w, upper_20_1w)
    lower_20_6h = align_htf_to_ltf(prices, df_1w, lower_20_1w)
    
    # Calculate 6h ATR(14) for volatility filter
    tr1 = high - low
    tr2 = np.abs(high - np.concatenate([[close[0]], close[:-1]]))
    tr3 = np.abs(low - np.concatenate([[close[0]], close[:-1]]))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    atr_14 = pd.Series(tr).ewm(span=14, adjust=False, min_periods=14).mean().values
    
    # Calculate 6h volume ratio (current vs 20-period average)
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_ratio = volume / (vol_ma_20 + 1e-10)
    
    signals = np.zeros(n)
    
    # Session filter: 00-24 UTC (always true for 6h, kept for structure)
    hours = prices.index.hour
    in_session = (hours >= 0) & (hours <= 23)
    
    for i in range(200, n):
        # Skip if any required data is NaN
        if (np.isnan(upper_20_6h[i]) or np.isnan(lower_20_6h[i]) or 
            np.isnan(ema200_1d_aligned[i]) or np.isnan(rsi_14_1d_aligned[i]) or 
            np.isnan(atr_14[i]) or np.isnan(volume_ratio[i]) or not in_session[i]):
            signals[i] = 0.0
            continue
        
        # Long conditions:
        # 1. 6h price breaks above weekly Donchian upper (20) - bullish breakout
        # 2. Price above 1d EMA200 (bullish trend filter)
        # 3. 1d RSI > 50 (bullish momentum)
        # 4. Volume confirmation: volume > 1.5x average
        # 5. Volatility filter: ATR > 0.3% of price (avoid low volatility chop)
        if (close[i] > upper_20_6h[i] and
            close[i] > ema200_1d_aligned[i] and
            rsi_14_1d_aligned[i] > 50 and
            volume_ratio[i] > 1.5 and
            atr_14[i] > 0.003 * close[i]):
            signals[i] = 0.25
            
        # Short conditions:
        # 1. 6h price breaks below weekly Donchian lower (20) - bearish breakdown
        # 2. Price below 1d EMA200 (bearish trend filter)
        # 3. 1d RSI < 50 (bearish momentum)
        # 4. Volume confirmation: volume > 1.5x average
        # 5. Volatility filter: ATR > 0.3% of price
        elif (close[i] < lower_20_6h[i] and
              close[i] < ema200_1d_aligned[i] and
              rsi_14_1d_aligned[i] < 50 and
              volume_ratio[i] > 1.5 and
              atr_14[i] > 0.003 * close[i]):
            signals[i] = -0.25
        else:
            signals[i] = 0.0
    
    return signals

name = "6h_1w_Donchian20_1d_EMA200_RSI_Volume_Filter_v1"
timeframe = "6h"
leverage = 1.0
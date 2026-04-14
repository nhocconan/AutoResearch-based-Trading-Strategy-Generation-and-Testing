#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Get 1d data for calculations
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate 1d ATR(14) for volatility filter and stop
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    tr1 = high_1d - low_1d
    tr2 = np.abs(high_1d - np.roll(close_1d, 1))
    tr3 = np.abs(low_1d - np.roll(close_1d, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]  # First period
    atr_14 = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    # Calculate 1d EMA50 for trend filter
    ema50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Calculate 12-period RSI on 1d close for momentum filter
    delta = pd.Series(close_1d).diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=12, min_periods=12).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=12, min_periods=12).mean()
    loss = np.where(loss == 0, 1e-10, loss)
    rs = gain / loss
    rsi_12 = 100 - (100 / (1 + rs))
    rsi_12_values = rsi_12.values
    
    # Align 1d indicators to 12h timeframe
    atr_14_aligned = align_htf_to_ltf(prices, df_1d, atr_14)
    ema50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema50_1d)
    rsi_12_aligned = align_htf_to_ltf(prices, df_1d, rsi_12_values)
    
    # Calculate 12h Donchian channels (20-period)
    high_series = pd.Series(high)
    low_series = pd.Series(low)
    donchian_high = high_series.rolling(window=20, min_periods=20).max().values
    donchian_low = low_series.rolling(window=20, min_periods=20).min().values
    
    signals = np.zeros(n)
    position = 0
    position_size = 0.25  # 25% position size
    
    # Start after enough data for calculations
    start = max(20, 14, 50, 12)  # Donchian, ATR, EMA, RSI
    
    for i in range(start, n):
        # Skip if any critical data is NaN
        if (np.isnan(donchian_high[i]) or np.isnan(donchian_low[i]) or 
            np.isnan(atr_14_aligned[i]) or np.isnan(ema50_1d_aligned[i]) or 
            np.isnan(rsi_12_aligned[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        atr = atr_14_aligned[i]
        ema50 = ema50_1d_aligned[i]
        rsi = rsi_12_aligned[i]
        
        if position == 0:
            # Long: breakout above Donchian high with uptrend filter (price > EMA50) and RSI not overbought
            if price > donchian_high[i] and price > ema50 and rsi < 70:
                position = 1
                signals[i] = position_size
            # Short: breakout below Donchian low with downtrend filter (price < EMA50) and RSI not oversold
            elif price < donchian_low[i] and price < ema50 and rsi > 30:
                position = -1
                signals[i] = -position_size
            else:
                signals[i] = 0.0
        elif position == 1:
            # Exit long: price closes below Donchian low or ATR-based stop
            if price < donchian_low[i] or price < ema50 - 1.5 * atr:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = position_size
        elif position == -1:
            # Exit short: price closes above Donchian high or ATR-based stop
            if price > donchian_high[i] or price > ema50 + 1.5 * atr:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -position_size
    
    return signals

name = "12h_1d_Donchian_EMA50_RSI_Filter"
timeframe = "12h"
leverage = 1.0
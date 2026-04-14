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
    
    # Load 1w data ONCE before loop
    df_1w = get_htf_data(prices, '1w')
    
    # Calculate weekly 20-period RSI for trend filter
    delta = np.diff(df_1w['close'], prepend=df_1w['close'][0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    avg_gain = pd.Series(gain).ewm(alpha=1/20, adjust=False, min_periods=20).mean().values
    avg_loss = pd.Series(loss).ewm(alpha=1/20, adjust=False, min_periods=20).mean().values
    rs = avg_gain / (avg_loss + 1e-10)
    rsi_20w = 100 - (100 / (1 + rs))
    rsi_20w_aligned = align_htf_to_ltf(prices, df_1w, rsi_20w)
    
    # Calculate weekly ATR(14) for volatility filter
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    tr1 = high_1w - low_1w
    tr2 = np.abs(high_1w - np.roll(close_1w, 1))
    tr3 = np.abs(low_1w - np.roll(close_1w, 1))
    tr2[0] = np.inf
    tr3[0] = np.inf
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    atr_1w = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    atr_1w_aligned = align_htf_to_ltf(prices, df_1w, atr_1w)
    
    # Calculate 6h Donchian(20) for breakout signals
    donch_high = pd.Series(high).rolling(window=20, min_periods=20).max().values
    donch_low = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    signals = np.zeros(n)
    position = 0
    position_size = 0.25  # 25% position size
    
    # Start after enough data for calculations
    start = 100
    
    for i in range(start, n):
        # Skip if any critical data is NaN
        if (np.isnan(rsi_20w_aligned[i]) or 
            np.isnan(atr_1w_aligned[i]) or
            np.isnan(donch_high[i]) or
            np.isnan(donch_low[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        
        # Weekly RSI filter: avoid extreme overbought/oversold conditions
        rsi_filter = (rsi_20w_aligned[i] > 30) & (rsi_20w_aligned[i] < 70)
        
        # Weekly volatility filter: ensure sufficient volatility for meaningful moves
        vol_filter = atr_1w_aligned[i] / price > 0.01 if price > 0 else False
        
        # 6h Donchian breakout conditions
        long_breakout = price > donch_high[i-1]  # Break above previous high
        short_breakout = price < donch_low[i-1]  # Break below previous low
        
        if position == 0:
            # Long entry: bullish breakout with favorable weekly conditions
            if long_breakout and rsi_filter and vol_filter:
                position = 1
                signals[i] = position_size
            # Short entry: bearish breakout with favorable weekly conditions
            elif short_breakout and rsi_filter and vol_filter:
                position = -1
                signals[i] = -position_size
            else:
                signals[i] = 0.0
        elif position == 1:
            # Exit long: price breaks below Donchian low or weekly RSI turns bearish
            if price < donch_low[i] or rsi_20w_aligned[i] < 40:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = position_size
        elif position == -1:
            # Exit short: price breaks above Donchian high or weekly RSI turns bullish
            if price > donch_high[i] or rsi_20w_aligned[i] > 60:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -position_size
    
    return signals

name = "6h_1wRSI20_ATR_Donchian20_Breakout_v1"
timeframe = "6h"
leverage = 1.0
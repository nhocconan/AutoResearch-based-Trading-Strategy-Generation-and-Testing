#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h Donchian breakout with 1d EMA200 trend filter and volume confirmation.
# Long when price breaks above Donchian(20) high, price > EMA200, and volume > 1.5x average.
# Short when price breaks below Donchian(20) low, price < EMA200, and volume > 1.5x average.
# Exit when price crosses EMA200 in opposite direction or volatility expands (ATR > 1.5x average).
# Uses discrete position size 0.25 to minimize churn. Target: 20-40 trades/year per symbol.
# Designed to capture trends in both bull and bear markets with strict entry filters.
name = "12h_Donchian20_1dEMA200_Volume_ATRFilter"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 210:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1d data for EMA200 trend filter
    df_1d = get_htf_data(prices, '1d')
    close_1d = df_1d['close'].values
    
    # Calculate EMA200 on daily
    ema_200_1d = pd.Series(close_1d).ewm(span=200, adjust=False, min_periods=200).mean().values
    
    # Calculate ATR(14) for volatility filter
    tr = np.zeros(n)
    for i in range(1, n):
        tr[i] = max(high[i] - low[i], abs(high[i] - close[i-1]), abs(low[i] - close[i-1]))
    atr_14 = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    atr_ma_50 = pd.Series(atr_14).rolling(window=50, min_periods=50).mean().values
    
    # Calculate Donchian channels (20-period)
    donchian_high = pd.Series(high).rolling(window=20, min_periods=20).max().values
    donchian_low = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # Align 1d EMA200 to 12h
    ema_200_aligned = align_htf_to_ltf(prices, df_1d, ema_200_1d)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(200, 50)  # Ensure EMA200 and ATR MA are ready
    
    for i in range(start_idx, n):
        # Skip if any required data is not available
        if (np.isnan(ema_200_aligned[i]) or np.isnan(donchian_high[i]) or 
            np.isnan(donchian_low[i]) or np.isnan(atr_ma_50[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        ema_200_val = ema_200_aligned[i]
        vol = volume[i]
        vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values[i]
        atr = atr_14[i]
        atr_ma = atr_ma_50[i]
        
        # Volume confirmation: current volume > 1.5x 20-period average
        volume_confirmed = vol > 1.5 * vol_ma_20
        
        # Volatility filter: avoid trading when volatility is too high
        vol_filter = atr < 1.5 * atr_ma
        
        if position == 0:
            # Enter long if price breaks above Donchian high, above EMA200, volume confirmed, and vol filter
            if (price > donchian_high[i] and price > ema_200_val and 
                volume_confirmed and vol_filter):
                signals[i] = 0.25
                position = 1
            # Enter short if price breaks below Donchian low, below EMA200, volume confirmed, and vol filter
            elif (price < donchian_low[i] and price < ema_200_val and 
                  volume_confirmed and vol_filter):
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit long when price crosses below EMA200 or breaks below Donchian low
            if price < ema_200_val or price < donchian_low[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short when price crosses above EMA200 or breaks above Donchian high
            if price > ema_200_val or price > donchian_high[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals
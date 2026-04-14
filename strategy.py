#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Donchian Channel Breakout with 12h EMA Trend Filter and Volume Confirmation
# Uses 4h Donchian(20) breakouts for entry signals in direction of 12h EMA(50) trend
# Volume confirmation (>1.3x average) ensures institutional participation
# ATR-based stop loss (2x ATR) manages risk
# Designed to work in both bull and bear markets by trading with the higher timeframe trend
# Target: 20-40 trades/year (80-160 total over 4 years) to minimize fee drag

def generate_signals(prices):
    n = len(prices)
    if n < 60:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Load 12h data ONCE before loop for EMA trend filter
    df_12h = get_htf_data(prices, '12h')
    
    # Calculate 12h EMA(50) for trend filter
    close_12h = df_12h['close'].values
    ema_12h = pd.Series(close_12h).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_12h_aligned = align_htf_to_ltf(prices, df_12h, ema_12h)
    
    # Calculate 4h ATR(14) for stop loss and volatility filter
    tr1 = high - low
    tr2 = np.abs(high - np.concatenate([[close[0]], closed]))
    tr3 = np.abs(low - np.concatenate([[close[0]], close[:-1]]))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    atr = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    # Calculate 4h Donchian Channel (20)
    donchian_high = pd.Series(high).rolling(window=20, min_periods=20).max().shift(1).values
    donchian_low = pd.Series(low).rolling(window=20, min_periods=20).min().shift(1).values
    
    # Volume confirmation: volume > 1.3x average volume (20-period)
    vol_series = pd.Series(volume)
    avg_vol = vol_series.rolling(window=20, min_periods=20).mean().shift(1).values
    
    signals = np.zeros(n)
    position = 0
    position_size = 0.25  # 25% position size
    
    # Start after enough data for calculations
    start = 50  # for EMA and Donchian calculations
    
    for i in range(start, n):
        # Skip if any critical data is NaN
        if (np.isnan(ema_12h_aligned[i]) or np.isnan(donchian_high[i]) or 
            np.isnan(donchian_low[i]) or np.isnan(atr[i]) or np.isnan(avg_vol[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        vol = volume[i]
        
        # Trend filter: only trade when price is above/below 12h EMA
        above_ema = price > ema_12h_aligned[i]
        below_ema = price < ema_12h_aligned[i]
        
        if position == 0:
            # Long: price breaks above Donchian high with volume filter and above EMA
            if price > donchian_high[i] and vol > 1.3 * avg_vol[i] and above_ema:
                position = 1
                signals[i] = position_size
            # Short: price breaks below Donchian low with volume filter and below EMA
            elif price < donchian_low[i] and vol > 1.3 * avg_vol[i] and below_ema:
                position = -1
                signals[i] = -position_size
            else:
                signals[i] = 0.0
        elif position == 1:
            # Exit long: price breaks below Donchian low or 2x ATR stop loss
            if price < donchian_low[i] or price < (signals[i-1] * position_size * 0 + close[i-1] - 2 * atr[i]):
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = position_size
        elif position == -1:
            # Exit short: price breaks above Donchian high or 2x ATR stop loss
            if price > donchian_high[i] or price > (signals[i-1] * position_size * 0 + close[i-1] + 2 * atr[i]):
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -position_size
    
    return signals

name = "4h_Donchian_Breakout_12hEMA_Volume"
timeframe = "4h"
leverage = 1.0
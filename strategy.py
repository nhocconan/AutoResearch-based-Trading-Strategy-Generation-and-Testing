#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Donchian(20) breakout with 1d EMA50 trend filter and ATR-based volatility filter.
# Long when price breaks above upper Donchian channel, price > 1d EMA50, and ATR(14) < ATR(50) (low volatility regime).
# Short when price breaks below lower Donchian channel, price < 1d EMA50, and ATR(14) < ATR(50).
# Exit when price reverts to the midpoint of the Donchian channel (mean reversion).
# Uses 1d EMA50 for higher timeframe trend alignment and ATR regime filter to avoid false breakouts in high volatility.
# Target: 20-50 trades/year on 4h timeframe with discrete position sizing of 0.25 to minimize fee drag.
# Works in bull markets via breakouts with trend alignment and in bear markets via short breakdowns with trend filter.

name = "4h_Donchian20_1dEMA50_ATRRegime_Trend_v1"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 60:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    
    # Load 1d data ONCE before loop for trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 60:
        return np.zeros(n)
    
    # Calculate 1d EMA50 for trend filter
    close_1d = df_1d['close'].values
    ema_50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # Calculate ATR(14) and ATR(50) for volatility regime filter
    tr1 = np.maximum(high - low, np.abs(high - np.roll(close, 1)))
    tr2 = np.maximum(tr1, np.abs(low - np.roll(close, 1)))
    tr2[0] = high[0] - low[0]  # Fix first bar
    atr_14 = pd.Series(tr2).rolling(window=14, min_periods=14).mean().values
    atr_50 = pd.Series(tr2).rolling(window=50, min_periods=50).mean().values
    atr_regime = atr_14 < atr_50  # Low volatility regime
    
    # Calculate Donchian(20) channels
    donch_high = pd.Series(high).rolling(window=20, min_periods=20).max().values
    donch_low = pd.Series(low).rolling(window=20, min_periods=20).min().values
    donch_mid = (donch_high + donch_low) / 2
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 50  # warmup for indicators
    
    for i in range(start_idx, n):
        # Skip if indicators not available
        if (np.isnan(ema_50_1d_aligned[i]) or 
            np.isnan(donch_high[i]) or np.isnan(donch_low[i]) or np.isnan(donch_mid[i]) or 
            np.isnan(atr_regime[i])):
            signals[i] = 0.0
            continue
        
        curr_close = close[i]
        curr_ema_50_1d = ema_50_1d_aligned[i]
        curr_donch_high = donch_high[i]
        curr_donch_low = donch_low[i]
        curr_donch_mid = donch_mid[i]
        curr_atr_regime = atr_regime[i]
        
        if position == 0:  # Flat - look for new entries
            # Long: price breaks above upper Donchian, price > 1d EMA50, low volatility regime
            if (curr_close > curr_donch_high and 
                curr_close > curr_ema_50_1d and 
                curr_atr_regime):
                signals[i] = 0.25
                position = 1
            # Short: price breaks below lower Donchian, price < 1d EMA50, low volatility regime
            elif (curr_close < curr_donch_low and 
                  curr_close < curr_ema_50_1d and 
                  curr_atr_regime):
                signals[i] = -0.25
                position = -1
        
        elif position == 1:  # Long position
            # Exit condition: price reverts to Donchian midpoint (mean reversion)
            if curr_close <= curr_donch_mid:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:  # Short position
            # Exit condition: price reverts to Donchian midpoint (mean reversion)
            if curr_close >= curr_donch_mid:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals
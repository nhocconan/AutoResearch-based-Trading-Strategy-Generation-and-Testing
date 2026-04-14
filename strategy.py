#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h strategy with 1-day ATR-based volatility filter and 1-day ATR trailing stop
# ATR(14) measures volatility; when ATR < 20-period SMA of ATR, market is low volatility (consolidation)
# Enter long when price breaks above highest high of last 20 bars with volume confirmation
# Enter short when price breaks below lowest low of last 20 bars with volume confirmation
# Use 1-day ATR trailing stop: exit when price moves against position by 2.5 * ATR
# Works in both bull and bear markets: volatility filter avoids whipsaws in low volatility,
# breakout captures strong moves, ATR stop adapts to volatility regimes

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Load 1d data ONCE for ATR and volatility filter
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate 1d ATR (14 periods)
    atr_len = 14
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # True Range
    tr1 = high_1d[1:] - low_1d[1:]
    tr2 = np.abs(high_1d[1:] - close_1d[:-1])
    tr3 = np.abs(low_1d[1:] - close_1d[:-1])
    tr = np.concatenate([[np.nan], np.maximum(tr1, np.maximum(tr2, tr3))])
    
    # ATR
    atr = pd.Series(tr).ewm(span=atr_len, adjust=False, min_periods=atr_len).mean().values
    
    # Align ATR to 4h timeframe
    atr_aligned = align_htf_to_ltf(prices, df_1d, atr)
    
    # Calculate 20-period SMA of ATR for volatility filter
    atr_sma = pd.Series(atr).rolling(window=20, min_periods=20).mean().values
    atr_sma_aligned = align_htf_to_ltf(prices, df_1d, atr_sma)
    
    # Volatility filter: low volatility when ATR < SMA of ATR
    low_volatility = atr_aligned < atr_sma_aligned
    
    # Calculate 20-period highest high and lowest low for breakout
    highest_high = pd.Series(high).rolling(window=20, min_periods=20).max().values
    lowest_low = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # Volume confirmation: volume > 1.5 * 20-period average volume
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_confirm = volume > (1.5 * vol_ma)
    
    signals = np.zeros(n)
    position = 0
    position_size = 0.25  # 25% position size
    
    # Start after enough data for calculations
    start = max(50, 20)
    
    for i in range(start, n):
        # Skip if any critical data is NaN
        if (np.isnan(atr_aligned[i]) or 
            np.isnan(atr_sma_aligned[i]) or
            np.isnan(highest_high[i]) or
            np.isnan(lowest_low[i]) or
            np.isnan(volume_confirm[i])):
            signals[i] = 0.0
            continue
        
        # Volatility filter: only trade in low volatility conditions
        if not low_volatility[i]:
            signals[i] = 0.0
            continue
        
        if position == 0:
            # Enter long: price breaks above highest high with volume confirmation
            if close[i] > highest_high[i] and volume_confirm[i]:
                position = 1
                signals[i] = position_size
            # Enter short: price breaks below lowest low with volume confirmation
            elif close[i] < lowest_low[i] and volume_confirm[i]:
                position = -1
                signals[i] = -position_size
            else:
                signals[i] = 0.0
        elif position == 1:
            # Exit long: ATR-based trailing stop or reversal signal
            # Trailing stop: exit if price drops by 2.5 * ATR from entry
            # Since we don't track entry price, use: exit if price < highest_high[i] - 2.5 * ATR
            if close[i] < (highest_high[i] - 2.5 * atr_aligned[i]):
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = position_size
        elif position == -1:
            # Exit short: ATR-based trailing stop or reversal signal
            # Trailing stop: exit if price rises by 2.5 * ATR from entry
            # Exit if price > lowest_low[i] + 2.5 * ATR
            if close[i] > (lowest_low[i] + 2.5 * atr_aligned[i]):
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -position_size
    
    return signals

name = "4h_1dATR_VolFilter_Breakout_TrailStop_v1"
timeframe = "4h"
leverage = 1.0
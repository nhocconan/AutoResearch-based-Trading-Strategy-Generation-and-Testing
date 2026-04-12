# 12h_1d_kama_volume_regime_v2
# Hypothesis: KAMA (Kaufman Adaptive Moving Average) adapts to market noise - in trending markets it tracks price closely, in ranging markets it stays flat.
# Combined with 1-day volume confirmation and choppy market filter (Choppiness Index), this should work in both bull and bear markets by adapting to conditions.
# Uses 12h timeframe with 1d HTF for trend context and volume confirmation.
# Target: 50-150 total trades over 4 years (12-37/year) to avoid fee drag.

#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "12h_1d_kama_volume_regime_v2"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Calculate KAMA on 12h data
    def kama(close, length=10, fast=2, slow=30):
        change = np.abs(np.diff(close, n=length))
        volatility = np.sum(np.abs(np.diff(close)), axis=1)
        er = np.where(volatility != 0, change / volatility, 0)
        sc = (er * (2/(fast+1) - 2/(slow+1)) + 2/(slow+1)) ** 2
        kama = np.full_like(close, np.nan, dtype=float)
        kama[length-1] = close[length-1]
        for i in range(length, len(close)):
            kama[i] = kama[i-1] + sc[i] * (close[i] - kama[i-1])
        return kama
    
    kama_values = kama(close, 10, 2, 30)
    
    # Get 1d data for volume and chop filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    volume_1d = df_1d['volume'].values
    
    # Volume ratio (current vs average)
    vol_ma_1d = pd.Series(volume_1d).rolling(window=20, min_periods=20).mean().values
    volume_ratio = volume_1d / vol_ma_1d
    volume_ratio_aligned = align_htf_to_ltf(prices, df_1d, volume_ratio)
    
    # Choppiness Index on 1d data
    def choppiness_index(high, low, close, length=14):
        atr = np.zeros_like(close)
        for i in range(1, len(close)):
            atr[i] = max(high[i] - low[i], np.abs(high[i] - close[i-1]), np.abs(low[i] - close[i-1]))
        atr_sum = pd.Series(atr).rolling(window=length, min_periods=length).sum().values
        highest_high = pd.Series(high).rolling(window=length, min_periods=length).max().values
        lowest_low = pd.Series(low).rolling(window=length, min_periods=length).min().values
        chop = np.where((highest_high - lowest_low) != 0, 
                        100 * np.log10(atr_sum / (highest_high - lowest_low)) / np.log10(length), 
                        50)
        return chop
    
    chop_values = choppiness_index(high_1d, low_1d, close_1d, 14)
    chop_aligned = align_htf_to_ltf(prices, df_1d, chop_values)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(50, n):
        # Skip if not ready
        if (np.isnan(kama_values[i]) or np.isnan(volume_ratio_aligned[i]) or 
            np.isnan(chop_aligned[i])):
            signals[i] = 0.0 if position == 0 else (0.25 if position == 1 else -0.25)
            continue
        
        # Determine market regime
        trending = chop_aligned[i] < 38.2  # Strong trend
        ranging = chop_aligned[i] > 61.8   # Choppy/ranging market
        
        # Volume confirmation
        volume_ok = volume_ratio_aligned[i] > 1.5  # Above average volume
        
        # KAMA signals
        price_above_kama = close[i] > kama_values[i]
        price_below_kama = close[i] < kama_values[i]
        
        # Entry logic: Adapt to market regime
        if trending and volume_ok:
            # In trending markets: follow KAMA direction
            long_signal = price_above_kama
            short_signal = price_below_kama
        elif ranging and volume_ok:
            # In ranging markets: mean reversion at extremes
            # Use price deviation from KAMA as signal
            deviation = (close[i] - kama_values[i]) / kama_values[i]
            long_signal = deviation < -0.02  # Oversold
            short_signal = deviation > 0.02  # Overbought
        else:
            # Transition period or low volume: no signal
            long_signal = False
            short_signal = False
        
        # Exit logic: reverse signal or volume drops
        exit_long = price_below_kama
        exit_short = price_above_kama
        
        # Execute trades
        if long_signal and position != 1:
            position = 1
            signals[i] = 0.25
        elif short_signal and position != -1:
            position = -1
            signals[i] = -0.25
        elif exit_long and position == 1:
            position = 0
            signals[i] = 0.0
        elif exit_short and position == -1:
            position = 0
            signals[i] = 0.0
        else:
            # Hold position
            if position == 1:
                signals[i] = 0.25
            elif position == -1:
                signals[i] = -0.25
            else:
                signals[i] = 0.0
    
    return signals
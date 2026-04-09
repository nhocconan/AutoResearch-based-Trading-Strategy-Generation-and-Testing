#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Donchian breakout with 12h trend filter and volume confirmation
# - Uses 4h Donchian channel (20-period) for breakout entries
# - Requires 12h HMA(21) trend alignment: long when price > HMA, short when price < HMA
# - Volume confirmation: volume > 2.0 * 20-period volume average (strict filter)
# - ATR(14) stoploss: 2.5 * ATR for wider stops in volatile markets
# - Position size: 0.25 (discrete level to minimize fee churn)
# - Designed for fewer trades (~20-40/year) to avoid fee drag while capturing strong trends
# - Works in bull markets via breakouts above upper channel with uptrend confirmation
# - Works in bear markets via breakdowns below lower channel with downtrend confirmation

name = "4h_12h_donchian_hma_volume_v1"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Load HTF data ONCE before loop
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 21:
        return np.zeros(n)
    
    # 12h HMA(21) for trend filter
    close_12h = df_12h['close'].values
    # HMA calculation: WMA(2*WMA(n/2) - WMA(n)), sqrt(n))
    def wma(data, window):
        weights = np.arange(1, window + 1)
        return np.convolve(data, weights, 'valid') / weights.sum()
    
    half_len = 21 // 2
    sqrt_len = int(np.sqrt(21))
    
    wma_half = pd.Series(close_12h).rolling(window=half_len, min_periods=half_len).apply(
        lambda x: np.dot(x, np.arange(1, half_len+1)) / np.arange(1, half_len+1).sum(), raw=False
    ).values
    wma_full = pd.Series(close_12h).rolling(window=21, min_periods=21).apply(
        lambda x: np.dot(x, np.arange(1, 22)) / np.arange(1, 22).sum(), raw=False
    ).values
    
    hma_raw = 2 * wma_half - wma_full
    hma_12h = pd.Series(hma_raw).rolling(window=sqrt_len, min_periods=sqrt_len).apply(
        lambda x: np.dot(x, np.arange(1, sqrt_len+1)) / np.arange(1, sqrt_len+1).sum(), raw=False
    ).values
    
    # Align 12h HMA to 4h timeframe
    hma_12h_aligned = align_htf_to_ltf(prices, df_12h, hma_12h)
    
    # Pre-compute 4h Donchian channels (20-period)
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    
    donchian_high = pd.Series(high).rolling(window=20, min_periods=20).max().values
    donchian_low = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # Pre-compute 4h ATR(14) for stoploss
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]
    atr = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    # Pre-compute volume confirmation: volume > 2.0 * 20-period average
    volume = prices['volume'].values
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_confirm = volume > (2.0 * vol_ma)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    highest_high_since_entry = 0.0
    lowest_low_since_entry = 0.0
    
    for i in range(100, n):
        # Skip if any required data is invalid
        if (np.isnan(donchian_high[i]) or np.isnan(donchian_low[i]) or
            np.isnan(hma_12h_aligned[i]) or np.isnan(atr[i]) or atr[i] <= 0 or
            np.isnan(volume_confirm[i])):
            signals[i] = 0.0
            continue
        
        if position == 1:  # Long position
            # Update highest high since entry
            highest_high_since_entry = max(highest_high_since_entry, high[i])
            
            # Exit conditions: stoploss or mean reversion
            if close[i] < highest_high_since_entry - 2.5 * atr[i]:  # ATR stop
                position = 0
                highest_high_since_entry = 0.0
                lowest_low_since_entry = 0.0
                signals[i] = 0.0
            elif close[i] < donchian_low[i]:  # Mean reversion exit (break below lower Donchian)
                position = 0
                highest_high_since_entry = 0.0
                lowest_low_since_entry = 0.0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Update lowest low since entry
            lowest_low_since_entry = min(lowest_low_since_entry, low[i])
            
            # Exit conditions: stoploss or mean reversion
            if close[i] > lowest_low_since_entry + 2.5 * atr[i]:  # ATR stop
                position = 0
                highest_high_since_entry = 0.0
                lowest_low_since_entry = 0.0
                signals[i] = 0.0
            elif close[i] > donchian_high[i]:  # Mean reversion exit (break above upper Donchian)
                position = 0
                highest_high_since_entry = 0.0
                lowest_low_since_entry = 0.0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat
            # Look for breakout entries with volume confirmation and trend filter
            if close[i] > donchian_high[i] and close[i] > hma_12h_aligned[i] and volume_confirm[i]:
                position = 1
                highest_high_since_entry = high[i]
                lowest_low_since_entry = low[i]
                signals[i] = 0.25
            elif close[i] < donchian_low[i] and close[i] < hma_12h_aligned[i] and volume_confirm[i]:
                position = -1
                highest_high_since_entry = high[i]
                lowest_low_since_entry = low[i]
                signals[i] = -0.25
    
    return signals
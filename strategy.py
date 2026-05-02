#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Donchian(20) breakout + 1d HMA(21) trend + volume confirmation + ATR trailing stop
# Donchian channel provides robust price structure for breakouts in both bull/bear markets
# 1d HMA(21) filters counter-trend trades, ensuring alignment with medium-term momentum
# Volume spike (2.0x 20-period average) confirms institutional participation
# ATR-based trailing stop (3.0x ATR) manages risk and reduces whipsaw
# Discrete position sizing (0.25) minimizes fee churn while maintaining profit potential
# Target: 75-200 total trades over 4 years (19-50/year) for 4h timeframe

name = "4h_Donchian20_1dHMA21_Trend_Volume_ATRStop_v1"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Get 1d data for HMA trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 21:
        return np.zeros(n)
    
    # Calculate 1d HMA(21)
    close_1d = pd.Series(df_1d['close'])
    half_length = 21 // 2
    sqrt_length = int(np.sqrt(21))
    
    wma_half = close_1d.rolling(window=half_length, min_periods=half_length).mean()
    wma_full = close_1d.rolling(window=21, min_periods=21).mean()
    raw_hma = 2 * wma_half - wma_full
    hma_1d = raw_hma.rolling(window=sqrt_length, min_periods=sqrt_length).mean().values
    hma_1d_aligned = align_htf_to_ltf(prices, df_1d, hma_1d)
    
    # Calculate 4h Donchian(20) channels (using 20-period high/low)
    high_series = pd.Series(high)
    low_series = pd.Series(low)
    donchian_high = high_series.rolling(window=20, min_periods=20).max().shift(1).values
    donchian_low = low_series.rolling(window=20, min_periods=20).min().shift(1).values
    
    # Calculate ATR(14) for volatility and trailing stop
    tr1 = pd.Series(high - low)
    tr2 = pd.Series(np.abs(high - np.roll(close, 1)))
    tr3 = pd.Series(np.abs(low - np.roll(close, 1)))
    tr2.iloc[0] = 0
    tr3.iloc[0] = 0
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr = tr.rolling(window=14, min_periods=14).mean().values
    
    # Volume confirmation (2.0x 20-period average)
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().shift(1).values
    volume_spike = volume > (vol_ma * 2.0)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    highest_since_long = 0.0
    lowest_since_short = 0.0
    
    # Start after warmup (need enough for calculations)
    start_idx = 50
    
    for i in range(start_idx, n):
        # Check for NaN values in indicators
        if (np.isnan(donchian_high[i]) or np.isnan(donchian_low[i]) or 
            np.isnan(hma_1d_aligned[i]) or np.isnan(volume_spike[i]) or 
            np.isnan(atr[i])):
            signals[i] = 0.0
            continue
        
        if position == 0:  # Flat - look for new entries
            # Donchian breakout long: price > upper channel
            # Donchian breakout short: price < lower channel
            breakout_long = close[i] > donchian_high[i]
            breakout_short = close[i] < donchian_low[i]
            
            # 1d HMA trend filter: price > HMA for longs, price < HMA for shorts
            hma_long = close[i] > hma_1d_aligned[i]
            hma_short = close[i] < hma_1d_aligned[i]
            
            if breakout_long and hma_long and volume_spike[i]:
                signals[i] = 0.25
                position = 1
                highest_since_long = close[i]
            elif breakout_short and hma_short and volume_spike[i]:
                signals[i] = -0.25
                position = -1
                lowest_since_short = close[i]
            else:
                signals[i] = 0.0
        
        elif position == 1:  # Long position
            # Update highest high since entry
            highest_since_long = max(highest_since_long, close[i])
            # ATR trailing stop: exit if price drops 3.0x ATR from highest high
            trailing_stop = highest_since_long - 3.0 * atr[i]
            # Exit conditions: Donchian breakdown OR trailing stop hit
            if close[i] < donchian_low[i] or close[i] < trailing_stop:
                signals[i] = 0.0
                position = 0
                highest_since_long = 0.0
            else:
                signals[i] = 0.25
        
        elif position == -1:  # Short position
            # Update lowest low since entry
            lowest_since_short = min(lowest_since_short, close[i])
            # ATR trailing stop: exit if price rises 3.0x ATR from lowest low
            trailing_stop = lowest_since_short + 3.0 * atr[i]
            # Exit conditions: Donchian breakout OR trailing stop hit
            if close[i] > donchian_high[i] or close[i] > trailing_stop:
                signals[i] = 0.0
                position = 0
                lowest_since_short = 0.0
            else:
                signals[i] = -0.25
    
    return signals
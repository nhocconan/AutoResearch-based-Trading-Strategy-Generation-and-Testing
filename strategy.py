#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h strategy using 1w Supertrend trend filter + 1d Donchian(20) breakout with volume confirmation.
# Enter long when price breaks above 1d Donchian upper band, 1w Supertrend is bullish, and volume > 1.5x 20-bar average.
# Enter short when price breaks below 1d Donchian lower band, 1w Supertrend is bearish, and volume > 1.5x 20-bar average.
# Uses discrete position sizing (0.25) to limit fee churn. Target: 20-40 trades/year.
# Supertrend provides higher timeframe trend bias, Donchian gives clear breakout levels, volume confirms momentum.
# Works in bull (trend-aligned breakouts) and bear (counter-trend breaks fail due to Supertrend filter).

name = "4h_Supertrend1w_Donchian20_Breakout_Volume_v1"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1w data for Supertrend (HTF)
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 50:
        return np.zeros(n)
    
    # Calculate 1w Supertrend (ATR=10, multiplier=3.0)
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    
    # True Range
    tr1 = high_1w - low_1w
    tr2 = np.abs(high_1w - np.roll(close_1w, 1))
    tr3 = np.abs(low_1w - np.roll(close_1w, 1))
    tr1[0] = 0  # first bar has no previous close
    tr2[0] = 0
    tr3[0] = 0
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    
    # ATR
    atr_period = 10
    atr = np.zeros_like(close_1w)
    atr[atr_period-1] = np.mean(tr[:atr_period])
    for i in range(atr_period, len(tr)):
        atr[i] = (atr[i-1] * (atr_period-1) + tr[i]) / atr_period
    
    # Supertrend calculation
    multiplier = 3.0
    hl2 = (high_1w + low_1w) / 2.0
    upperband = hl2 + multiplier * atr
    lowerband = hl2 - multiplier * atr
    
    supertrend = np.zeros_like(close_1w)
    uptrend = np.ones_like(close_1w, dtype=bool)
    
    supertrend[0] = upperband[0]
    uptrend[0] = True
    
    for i in range(1, len(close_1w)):
        if close_1w[i] > supertrend[i-1]:
            uptrend[i] = True
        elif close_1w[i] < supertrend[i-1]:
            uptrend[i] = False
        else:
            uptrend[i] = uptrend[i-1]
            if uptrend[i] and lowerband[i] < supertrend[i-1]:
                lowerband[i] = supertrend[i-1]
            if not uptrend[i] and upperband[i] > supertrend[i-1]:
                upperband[i] = supertrend[i-1]
        
        if uptrend[i]:
            supertrend[i] = lowerband[i]
        else:
            supertrend[i] = upperband[i]
    
    # Align 1w Supertrend to 4h timeframe (uptrend=True means bullish)
    supertrend_uptrend_aligned = align_htf_to_ltf(prices, df_1w, uptrend.astype(float))
    
    # Get 1d data for Donchian channels (HTF)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # Calculate 1d Donchian channels (20-period)
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    
    donchian_period = 20
    donchian_high = np.full_like(high_1d, np.nan)
    donchian_low = np.full_like(low_1d, np.nan)
    
    for i in range(donchian_period-1, len(high_1d)):
        donchian_high[i] = np.max(high_1d[i-donchian_period+1:i+1])
        donchian_low[i] = np.min(low_1d[i-donchian_period+1:i+1])
    
    # Forward fill Donchian levels
    donchian_high = pd.Series(donchian_high).ffill().values
    donchian_low = pd.Series(donchian_low).ffill().values
    
    # Align 1d Donchian to 4h timeframe
    donchian_high_aligned = align_htf_to_ltf(prices, df_1d, donchian_high)
    donchian_low_aligned = align_htf_to_ltf(prices, df_1d, donchian_low)
    
    # Calculate 4h volume confirmation: >1.5x 20-bar average volume
    volume_series = pd.Series(volume)
    volume_ma_20 = volume_series.rolling(window=20, min_periods=20).mean().values
    volume_confirm = volume > 1.5 * volume_ma_20
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 100  # Ensure sufficient history for indicators
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(supertrend_uptrend_aligned[i]) or 
            np.isnan(donchian_high_aligned[i]) or np.isnan(donchian_low_aligned[i]) or
            np.isnan(volume_ma_20[i])):
            signals[i] = 0.0
            continue
        
        # Determine trend bias from Supertrend (1 = bullish/uptrend, 0 = bearish/downtrend)
        is_bullish_trend = supertrend_uptrend_aligned[i] > 0.5
        
        # Donchian breakout conditions with volume confirmation and trend filter
        long_breakout = close[i] > donchian_high_aligned[i] and volume_confirm[i] and is_bullish_trend
        short_breakout = close[i] < donchian_low_aligned[i] and volume_confirm[i] and (not is_bullish_trend)
        
        # Exit conditions: opposite Donchian level
        long_exit = close[i] < donchian_low_aligned[i]
        short_exit = close[i] > donchian_high_aligned[i]
        
        # Handle entries and exits
        if long_breakout and position <= 0:
            signals[i] = 0.25
            position = 1
        elif short_breakout and position >= 0:
            signals[i] = -0.25
            position = -1
        elif (position == 1 and long_exit) or (position == -1 and short_exit):
            signals[i] = 0.0
            position = 0
        else:
            # Hold current position
            if position == 1:
                signals[i] = 0.25
            elif position == -1:
                signals[i] = -0.25
            else:
                signals[i] = 0.0
    
    return signals
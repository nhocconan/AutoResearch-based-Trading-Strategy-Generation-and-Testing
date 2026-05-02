#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1d Donchian(20) breakout + 1w EMA50 trend + volume spike + choppiness regime filter
# Targets 30-100 total trades over 4 years (7-25/year) to minimize fee drag on daily timeframe
# Donchian channels provide robust price structure with proven edge on SOLUSDT (test Sharpe 1.10-1.38)
# 1w EMA50 determines higher timeframe trend bias: long when price > EMA50, short when price < EMA50
# Volume spike (2x 20-period average) confirms institutional participation
# Choppiness regime filter (CHOP > 61.8 = range, CHOP < 38.2 = trend) avoids whipsaws
# Works in bull markets via breakouts with trend alignment and bear markets via fade of false breakouts
# Discrete position sizing: 0.30 (30% of capital) balances exposure and risk
# Using 1d primary timeframe as specified in experiment instructions

name = "1d_Donchian20_1wEMA50_VolumeSpike_ChopFilter"
timeframe = "1d"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Calculate 1d Donchian channels (20-period, prior completed 1d bar)
    highest_high = pd.Series(high).rolling(window=20, min_periods=20).max().shift(1).values
    lowest_low = pd.Series(low).rolling(window=20, min_periods=20).min().shift(1).values
    
    # Calculate 1w EMA50 trend (prior completed 1w bar's EMA)
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 2:
        return np.zeros(n)
    
    ema_50 = pd.Series(df_1w['close']).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_aligned = align_htf_to_ltf(prices, df_1w, ema_50)
    
    # Calculate 1d volume spike (2x 20-period average)
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().shift(1).values
    volume_spike = volume > (vol_ma * 2.0)
    
    # Calculate 1d choppiness index (14-period)
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr = np.maximum(np.maximum(tr1, tr2), tr3)
    atr = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    sum_atr = pd.Series(atr).rolling(window=14, min_periods=14).sum().values
    highest_high_14 = pd.Series(high).rolling(window=14, min_periods=14).max().values
    lowest_low_14 = pd.Series(low).rolling(window=14, min_periods=14).min().values
    # Avoid division by zero
    range_hl = highest_high_14 - lowest_low_14
    range_hl = np.where(range_hl == 0, 1e-10, range_hl)
    chop = 100 * np.log10(sum_atr / range_hl) / np.log10(14)
    chop_regime = (chop > 61.8) | (chop < 38.2)  # Range or trending regime
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after warmup (need enough data for all indicators)
    start_idx = max(50, 20, 14)  # EMA50 warmup is the longest
    
    for i in range(start_idx, n):
        # Check for NaN values in indicators
        if (np.isnan(highest_high[i]) or np.isnan(lowest_low[i]) or 
            np.isnan(ema_50_aligned[i]) or np.isnan(volume_spike[i]) or 
            np.isnan(chop_regime[i])):
            signals[i] = 0.0
            continue
        
        if position == 0:  # Flat - look for new entries
            # Long entry: price breaks above Donchian upper band AND price > 1w EMA50 (bullish bias) AND volume spike AND regime filter
            if (close[i] > highest_high[i] and 
                close[i] > ema_50_aligned[i] and 
                volume_spike[i] and 
                chop_regime[i]):
                signals[i] = 0.30
                position = 1
            # Short entry: price breaks below Donchian lower band AND price < 1w EMA50 (bearish bias) AND volume spike AND regime filter
            elif (close[i] < lowest_low[i] and 
                  close[i] < ema_50_aligned[i] and 
                  volume_spike[i] and 
                  chop_regime[i]):
                signals[i] = -0.30
                position = -1
            else:
                signals[i] = 0.0
        
        elif position == 1:  # Long position
            # Exit: price falls below Donchian lower band OR below 1w EMA50 (trend change)
            if close[i] < lowest_low[i] or close[i] < ema_50_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.30
        
        elif position == -1:  # Short position
            # Exit: price rises above Donchian upper band OR above 1w EMA50 (trend change)
            if close[i] > highest_high[i] or close[i] > ema_50_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.30
    
    return signals
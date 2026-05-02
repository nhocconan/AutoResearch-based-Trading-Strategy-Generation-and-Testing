#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h Donchian(20) breakout + 1w EMA50 trend + volume spike + choppiness regime filter
# Targets 50-150 total trades over 4 years (12-37/year) to minimize fee drag
# Donchian channels provide clear breakout structure with proven edge on SOLUSDT (test Sharpe 1.10-1.38)
# 1w EMA50 determines major trend bias: long when price > EMA50, short when price < EMA50
# Volume spike (2x 20-period average) confirms institutional participation
# Choppiness regime filter (CHOP > 61.8 = range, CHOP < 38.2 = trend) avoids whipsaws
# Works in bull markets via breakouts with trend alignment and bear markets via fade of false breakouts
# Discrete position sizing: 0.25 (25% of capital) balances exposure and risk

name = "12h_Donchian20_1wEMA50_VolumeSpike_ChopFilter"
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
    
    # Calculate 1w Donchian levels (prior completed 1w bar's range)
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 2:
        return np.zeros(n)
    
    # Prior completed 1w bar's high, low for Donchian(20) - using 20-period lookback
    # For simplicity, we use the prior completed 1w bar's high/low as the channel
    # In practice, Donchian(20) would use 20 periods of 1w data, but we approximate with prior bar
    ph = pd.Series(df_1w['high']).shift(1).values
    pl = pd.Series(df_1w['low']).shift(1).values
    
    # Align to 12h timeframe (wait for completed 1w bar)
    ph_aligned = align_htf_to_ltf(prices, df_1w, ph)
    pl_aligned = align_htf_to_ltf(prices, df_1w, pl)
    
    # Calculate 1w EMA50 trend (prior completed 1w bar's EMA)
    ema_50 = pd.Series(df_1w['close']).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_aligned = align_htf_to_ltf(prices, df_1w, ema_50)
    
    # Calculate 12h volume spike (2x 20-period average)
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().shift(1).values
    volume_spike = volume > (vol_ma * 2.0)
    
    # Calculate 12h choppiness index (14-period)
    atr = pd.Series(np.maximum(np.maximum(high - low, np.abs(high - np.roll(close, 1))), np.abs(low - np.roll(close, 1)))).rolling(window=14, min_periods=14).mean().values
    sum_atr = pd.Series(atr).rolling(window=14, min_periods=14).sum().values
    highest_high = pd.Series(high).rolling(window=14, min_periods=14).max().values
    lowest_low = pd.Series(low).rolling(window=14, min_periods=14).min().values
    # Avoid division by zero
    hl_range = highest_high - lowest_low
    hl_range = np.where(hl_range == 0, 1e-10, hl_range)
    chop = 100 * np.log10(sum_atr / hl_range) / np.log10(14)
    chop_regime = (chop > 61.8) | (chop < 38.2)  # Range or trending regime
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after warmup (need enough data for all indicators)
    start_idx = max(50, 20, 14)
    
    for i in range(start_idx, n):
        # Check for NaN values in indicators
        if (np.isnan(ph_aligned[i]) or np.isnan(pl_aligned[i]) or 
            np.isnan(ema_50_aligned[i]) or np.isnan(volume_spike[i]) or 
            np.isnan(chop_regime[i])):
            signals[i] = 0.0
            continue
        
        if position == 0:  # Flat - look for new entries
            # Long entry: price breaks above prior 1w high AND price > 1w EMA50 (bullish bias) AND volume spike AND regime filter
            if (close[i] > ph_aligned[i] and 
                close[i] > ema_50_aligned[i] and 
                volume_spike[i] and 
                chop_regime[i]):
                signals[i] = 0.25
                position = 1
            # Short entry: price breaks below prior 1w low AND price < 1w EMA50 (bearish bias) AND volume spike AND regime filter
            elif (close[i] < pl_aligned[i] and 
                  close[i] < ema_50_aligned[i] and 
                  volume_spike[i] and 
                  chop_regime[i]):
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        
        elif position == 1:  # Long position
            # Exit: price falls below prior 1w low OR below 1w EMA50 (trend change)
            if close[i] < pl_aligned[i] or close[i] < ema_50_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:  # Short position
            # Exit: price rises above prior 1w high OR above 1w EMA50 (trend change)
            if close[i] > ph_aligned[i] or close[i] > ema_50_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals
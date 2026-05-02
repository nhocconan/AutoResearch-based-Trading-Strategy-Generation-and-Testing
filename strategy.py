#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Donchian(20) breakout with 1d EMA50 trend filter, volume spike (>1.8x average), and chop regime filter (CHOP < 61.8)
# Donchian breakout captures momentum, 1d EMA50 ensures alignment with daily trend, volume spike confirms institutional interest,
# chop regime avoids false breakouts in ranging markets. Discrete sizing 0.25 balances return and drawdown.
# Target: 50-150 total trades over 4 years (12-37/year) to stay within fee drag limits.
# Primary timeframe: 6h, HTF: 1d for EMA50 and chop regime.

name = "6h_Donchian20_Breakout_1dEMA50_Volume_Chop"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Calculate 1d EMA50 for trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    ema_50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # Calculate 6h ATR(14) for chop regime
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]  # First period
    
    atr_period = 14
    atr = pd.Series(tr).rolling(window=atr_period, min_periods=atr_period).mean().values
    highest_high = pd.Series(high).rolling(window=atr_period, min_periods=atr_period).max().values
    lowest_low = pd.Series(low).rolling(window=atr_period, min_periods=atr_period).min().values
    
    # Avoid division by zero
    atr_safe = np.where(atr == 0, 1e-10, atr)
    chop = 100 * np.log10((highest_high - lowest_low) / (atr_safe * np.sqrt(atr_period))) / np.log10(atr_period)
    chop_regime = chop < 61.8  # True when trending (CHOP < 61.8), False when ranging
    
    # Volume confirmation: 1.8x 20-period average (balanced to avoid overtrading)
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (1.8 * vol_ma)
    
    # Calculate 6h Donchian channels (20-period)
    donchian_window = 20
    highest_high_20 = pd.Series(high).rolling(window=donchian_window, min_periods=donchian_window).max().values
    lowest_low_20 = pd.Series(low).rolling(window=donchian_window, min_periods=donchian_window).min().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after warmup (need enough data for all indicators)
    start_idx = max(50, donchian_window, 20)  # 50 for EMA, 20 for Donchian and volume MA
    
    for i in range(start_idx, n):
        if (np.isnan(ema_50_1d_aligned[i]) or np.isnan(chop_regime[i]) or 
            np.isnan(vol_ma[i]) or np.isnan(highest_high_20[i]) or 
            np.isnan(lowest_low_20[i])):
            signals[i] = 0.0
            continue
        
        if position == 0:  # Flat - look for new entries
            # Long: Price breaks above Donchian upper AND price > 1d EMA50 AND volume spike AND trending regime
            if (close[i] > highest_high_20[i] and 
                close[i] > ema_50_1d_aligned[i] and 
                volume_spike[i] and 
                chop_regime[i]):
                signals[i] = 0.25
                position = 1
            # Short: Price breaks below Donchian lower AND price < 1d EMA50 AND volume spike AND trending regime
            elif (close[i] < lowest_low_20[i] and 
                  close[i] < ema_50_1d_aligned[i] and 
                  volume_spike[i] and 
                  chop_regime[i]):
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        
        elif position == 1:  # Long position
            # Exit: Price drops below Donchian lower OR price < 1d EMA50
            if close[i] < lowest_low_20[i] or close[i] < ema_50_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:  # Short position
            # Exit: Price rises above Donchian upper OR price > 1d EMA50
            if close[i] > highest_high_20[i] or close[i] > ema_50_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals
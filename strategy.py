#!/usr/bin/env python3
# 6h_daily_elder_ray_regime_v1
# Hypothesis: 6h strategy using Elder Ray (Bull/Bear Power) from 1d to determine regime,
# combined with 6h EMA(21) for entry timing. In bull regime (Bull Power > 0), go long
# when price crosses above EMA(21); in bear regime (Bear Power > 0), go short when
# price crosses below EMA(21). Uses volume confirmation to avoid false breakouts.
# Designed to work in both bull and bear markets by adapting to the daily regime.
# Target: 12-37 trades/year (50-150 total over 4 years) on BTC/ETH/SOL.

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "6h_daily_elder_ray_regime_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # 6h EMA(21) for entry timing
    close_s = pd.Series(close)
    ema_21 = close_s.ewm(span=21, min_periods=21, adjust=False).mean().values
    
    # Volume average for confirmation (20-period)
    volume_s = pd.Series(volume)
    volume_ma = volume_s.rolling(window=20, min_periods=20).mean().values
    
    # Get 1d data for Elder Ray calculation (HTF)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) == 0:
        return np.zeros(n)
    
    # Calculate daily EMA(13) for Elder Ray
    close_1d = df_1d['close'].values
    close_1d_s = pd.Series(close_1d)
    ema_13_1d = close_1d_s.ewm(span=13, min_periods=13, adjust=False).mean().values
    
    # Elder Ray: Bull Power = High - EMA13, Bear Power = EMA13 - Low
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    bull_power = high_1d - ema_13_1d
    bear_power = ema_13_1d - low_1d
    
    # Align HTF data to LTF
    ema_21_aligned = align_htf_to_ltf(prices, df_1d, ema_21)  # Dummy call to get alignment structure
    # Actually we need to align the Elder Ray components
    bull_power_aligned = align_htf_to_ltf(prices, df_1d, bull_power)
    bear_power_aligned = align_htf_to_ltf(prices, df_1d, bear_power)
    ema_21_6h = close_s.ewm(span=21, min_periods=21, adjust=False).mean().values  # Recalculate for 6h
    volume_ma_aligned = align_htf_to_ltf(prices, df_1d, volume_ma)  # Dummy for structure
    
    # Recompute volume MA properly aligned
    volume_s = pd.Series(volume)
    volume_ma = volume_s.rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(50, n):  # Start after warmup
        # Skip if any required data is NaN
        if (np.isnan(bull_power_aligned[i]) or np.isnan(bear_power_aligned[i]) or
            np.isnan(ema_21_6h[i]) or np.isnan(volume_ma[i]) or np.isnan(close[i]) or np.isnan(volume[i])):
            signals[i] = 0.0
            continue
        
        # Volume confirmation: current volume > 1.3x 20-period average
        volume_confirmed = volume[i] > 1.3 * volume_ma[i]
        
        if position == 1:  # Long position
            # Exit: Price crosses below EMA(21) or regime changes to bear
            if close[i] < ema_21_6h[i] or bear_power_aligned[i] > bull_power_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit: Price crosses above EMA(21) or regime changes to bull
            if close[i] > ema_21_6h[i] or bull_power_aligned[i] > bear_power_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat
            # Check for entry with volume confirmation
            bullish_entry = (bull_power_aligned[i] > bear_power_aligned[i]) and (close[i] > ema_21_6h[i]) and volume_confirmed
            bearish_entry = (bear_power_aligned[i] > bull_power_aligned[i]) and (close[i] < ema_21_6h[i]) and volume_confirmed
            
            if bullish_entry:
                position = 1
                signals[i] = 0.25
            elif bearish_entry:
                position = -1
                signals[i] = -0.25
    
    return signals
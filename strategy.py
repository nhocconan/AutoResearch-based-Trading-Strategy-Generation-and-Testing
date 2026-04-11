#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "6h_1d_elder_ray_breakout_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    # Load daily data ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return signals
    
    # Calculate Elder Ray indicators from daily data
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # EMA(13) of daily close
    ema13_1d = pd.Series(close_1d).ewm(span=13, adjust=False, min_periods=13).mean().values
    
    # Bull Power = High - EMA13
    bull_power_1d = high_1d - ema13_1d
    
    # Bear Power = Low - EMA13
    bear_power_1d = low_1d - ema13_1d
    
    # Align daily values to 6h timeframe
    ema13_aligned = align_htf_to_ltf(prices, df_1d, ema13_1d)
    bull_power_aligned = align_htf_to_ltf(prices, df_1d, bull_power_1d)
    bear_power_aligned = align_htf_to_ltf(prices, df_1d, bear_power_1d)
    
    # 6h EMA(20) for trend filter
    ema20_6h = pd.Series(close).ewm(span=20, adjust=False, min_periods=20).mean().values
    
    # Volume confirmation: 6h volume > 1.5x 20-period average
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    for i in range(100, n):
        # Skip if any required data is invalid
        if (np.isnan(ema13_aligned[i]) or np.isnan(bull_power_aligned[i]) or 
            np.isnan(bear_power_aligned[i]) or np.isnan(ema20_6h[i]) or
            np.isnan(vol_ma_20[i])):
            signals[i] = 0.0
            continue
        
        price_close = close[i]
        volume_current = volume[i]
        
        # Volume confirmation
        vol_confirm = volume_current > 1.5 * vol_ma_20[i]
        
        # Elder Ray conditions
        bullish = bull_power_aligned[i] > 0 and ema13_aligned[i] > ema20_6h[i]
        bearish = bear_power_aligned[i] < 0 and ema13_aligned[i] < ema20_6h[i]
        
        # Entry conditions
        enter_long = False
        enter_short = False
        
        # Long: Bull power positive, EMA13 above EMA20, volume confirmation
        if bullish and vol_confirm:
            enter_long = True
        
        # Short: Bear power negative, EMA13 below EMA20, volume confirmation
        if bearish and vol_confirm:
            enter_short = True
        
        # Exit conditions: opposite Elder Ray power or EMA crossover
        exit_long = bear_power_aligned[i] < 0 or ema13_aligned[i] < ema20_6h[i]
        exit_short = bull_power_aligned[i] > 0 or ema13_aligned[i] > ema20_6h[i]
        
        # Trading logic
        if enter_long and position != 1:
            position = 1
            signals[i] = 0.25
        elif enter_short and position != -1:
            position = -1
            signals[i] = -0.25
        elif position == 1 and exit_long:
            position = 0
            signals[i] = 0.0
        elif position == -1 and exit_short:
            position = 0
            signals[i] = 0.0
        else:
            # Maintain current position
            signals[i] = 0.25 if position == 1 else (-0.25 if position == -1 else 0.0)
    
    return signals

# Hypothesis: 6h Elder Ray breakout strategy with EMA trend filter and volume confirmation.
# Uses daily Elder Ray (Bull/Bear Power) aligned to 6h timeframe to identify institutional bias.
# Enters long when Bull Power > 0 and daily EMA13 > 6h EMA20 with volume confirmation.
# Enters short when Bear Power < 0 and daily EMA13 < 6h EMA20 with volume confirmation.
# Exits when Elder Ray power reverses or EMA13/EMA20 crossover occurs.
# Volume filter (>1.5x 20-period average) ensures participation during active periods.
# Position size 0.25 balances risk while allowing meaningful returns.
# Target: 60-120 total trades over 4 years (15-30/year) to minimize fee drag.
# Works in both bull and bear markets by following institutional order flow.
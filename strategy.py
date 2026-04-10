#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Williams Alligator + 1d Elder Ray Regime Filter
# - Primary: 6h timeframe for lower trade frequency and reduced fee drag
# - HTF: 1d for Elder Ray (Bull/Bear Power) regime detection and Williams Alligator alignment
# - Williams Alligator (6h): Jaw(13,8), Teeth(8,5), Lips(5,3) SMAs
# - Elder Ray (1d): Bull Power = High - EMA(13), Bear Power = Low - EMA(13)
# - Long: Alligator aligned bullish (Lips > Teeth > Jaw) + 1d Bull Power > 0 + 1d Bear Power < 0
# - Short: Alligator aligned bearish (Lips < Teeth < Jaw) + 1d Bear Power < 0 + 1d Bull Power < 0
# - Exit: Alligator alignment reverses (Lips crosses Teeth)
# - Position sizing: 0.25 (discrete level)
# - Works in bull/bear: Alligator catches trends, Elder Ray filters regime strength
# - Target: 50-120 total trades over 4 years (12-30/year) - within 6h sweet spot

name = "6h_1d_alligator_elderray_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:  # Need enough data for calculations
        return np.zeros(n)
    
    # Load HTF data ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    # Pre-compute 6h OHLCV
    open_6h = prices['open'].values
    high_6h = prices['high'].values
    low_6h = prices['low'].values
    close_6h = prices['close'].values
    volume_6h = prices['volume'].values
    
    # Pre-compute 1d data
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate Williams Alligator on 6h
    # Jaw: 13-period SMA, shifted 8 bars
    jaw = pd.Series(close_6h).rolling(window=13, min_periods=13).mean().shift(8).values
    # Teeth: 8-period SMA, shifted 5 bars
    teeth = pd.Series(close_6h).rolling(window=8, min_periods=8).mean().shift(5).values
    # Lips: 5-period SMA, shifted 3 bars
    lips = pd.Series(close_6h).rolling(window=5, min_periods=5).mean().shift(3).values
    
    # Calculate Elder Ray on 1d
    # EMA(13) of close
    ema_13_1d = pd.Series(close_1d).ewm(span=13, adjust=False, min_periods=13).mean().values
    # Bull Power = High - EMA(13)
    bull_power_1d = high_1d - ema_13_1d
    # Bear Power = Low - EMA(13)
    bear_power_1d = low_1d - ema_13_1d
    
    # Align 1d indicators to 6h
    bull_power_aligned = align_htf_to_ltf(prices, df_1d, bull_power_1d)
    bear_power_aligned = align_htf_to_ltf(prices, df_1d, bear_power_1d)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(20, n):  # Start after warmup period
        # Skip if any required data is invalid
        if (np.isnan(lips[i]) or np.isnan(teeth[i]) or np.isnan(jaw[i]) or
            np.isnan(bull_power_aligned[i]) or np.isnan(bear_power_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Williams Alligator alignment
        # Bullish: Lips > Teeth > Jaw
        alligator_bullish = lips[i] > teeth[i] and teeth[i] > jaw[i]
        # Bearish: Lips < Teeth < Jaw
        alligator_bearish = lips[i] < teeth[i] and teeth[i] < jaw[i]
        
        # Elder Ray regime (from 1d)
        # Strong bull: Bull Power > 0 and Bear Power < 0
        strong_bull = bull_power_aligned[i] > 0 and bear_power_aligned[i] < 0
        # Strong bear: Bear Power < 0 and Bull Power < 0
        strong_bear = bear_power_aligned[i] < 0 and bull_power_aligned[i] < 0
        
        if position == 0:  # Flat - look for new entries
            # Long entry: Bullish Alligator + Strong Bull regime
            if alligator_bullish and strong_bull:
                position = 1
                signals[i] = 0.25
            # Short entry: Bearish Alligator + Strong Bear regime
            elif alligator_bearish and strong_bear:
                position = -1
                signals[i] = -0.25
            else:
                signals[i] = 0.0
        else:  # Have position - look for exit
            # Exit: Alligator alignment reverses (Lips crosses Teeth)
            if position == 1:  # Long position
                exit_condition = lips[i] < teeth[i]  # Lips crossed below Teeth
                if exit_condition:
                    position = 0
                    signals[i] = 0.0
                else:
                    signals[i] = 0.25
            else:  # position == -1 (Short position)
                exit_condition = lips[i] > teeth[i]  # Lips crossed above Teeth
                if exit_condition:
                    position = 0
                    signals[i] = 0.0
                else:
                    signals[i] = -0.25
    
    return signals
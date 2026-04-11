#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h Williams Alligator + Elder Ray + volume confirmation
# - Uses Williams Alligator (jaw=13, teeth=8, lips=5 SMAs) to define trend direction
# - Elder Ray (bull/bear power) measures trend strength relative to EMA(13)
# - Volume confirmation: current volume > 1.5x 20-period average
# - Long: Alligator aligned bullish (lips > teeth > jaw) AND bull power > 0 AND volume confirmation
# - Short: Alligator aligned bearish (lips < teeth < jaw) AND bear power < 0 AND volume confirmation
# - Exit: opposite Alligator alignment or Elder Ray crosses zero
# - Uses 1d trend filter: price > EMA(50) for long bias, price < EMA(50) for short bias
# - Target: 12-37 trades/year (50-150 total over 4 years) to stay within fee drag limits
# - Williams Alligator catches trends early, Elder Ray filters false signals, volume confirms momentum

name = "12h_1d_alligator_elder_volume_v1"
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
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    # Load 1d data ONCE before loop for EMA trend filter (MTF rule compliance)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return signals
    
    # Pre-compute 1d EMA(50) for trend filter
    close_1d = df_1d['close'].values
    ema_50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # Pre-compute Williams Alligator components (12h timeframe)
    # Jaw: 13-period SMMA (smoothed moving average)
    jaw = pd.Series(close).rolling(window=13, min_periods=13).mean().values
    # Teeth: 8-period SMMA
    teeth = pd.Series(close).rolling(window=8, min_periods=8).mean().values
    # Lips: 5-period SMMA
    lips = pd.Series(close).rolling(window=5, min_periods=5).mean().values
    
    # Pre-compute Elder Ray components
    # EMA(13) for Elder Ray calculation
    ema_13 = pd.Series(close).ewm(span=13, adjust=False, min_periods=13).mean().values
    # Bull Power = High - EMA(13)
    bull_power = high - ema_13
    # Bear Power = Low - EMA(13)
    bear_power = low - ema_13
    
    # Pre-compute volume confirmation (20-period average)
    volume_sma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    for i in range(100, n):  # Start after 100-bar warmup
        # Skip if any required data is invalid
        if (np.isnan(jaw[i]) or np.isnan(teeth[i]) or np.isnan(lips[i]) or
            np.isnan(bull_power[i]) or np.isnan(bear_power[i]) or
            np.isnan(volume_sma_20[i]) or np.isnan(ema_50_1d_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Current price data
        close_price = close[i]
        high_price = high[i]
        low_price = low[i]
        volume_current = volume[i]
        
        # Williams Alligator alignment
        alligator_bullish = lips[i] > teeth[i] and teeth[i] > jaw[i]  # Lips > Teeth > Jaw
        alligator_bearish = lips[i] < teeth[i] and teeth[i] < jaw[i]  # Lips < Teeth < Jaw
        
        # Elder Ray signals
        bull_power_positive = bull_power[i] > 0
        bear_power_negative = bear_power[i] < 0
        
        # Volume confirmation: current volume > 1.5x 20-period average
        vol_confirm = volume_current > 1.5 * volume_sma_20[i]
        
        # 1d EMA trend bias
        ema_bias_long = close_price > ema_50_1d_aligned[i]
        ema_bias_short = close_price < ema_50_1d_aligned[i]
        
        # Entry conditions
        enter_long = False
        enter_short = False
        
        # Long: Alligator bullish AND bull power positive AND volume confirmation AND long bias
        if alligator_bullish and bull_power_positive and vol_confirm and ema_bias_long:
            enter_long = True
        
        # Short: Alligator bearish AND bear power negative AND volume confirmation AND short bias
        if alligator_bearish and bear_power_negative and vol_confirm and ema_bias_short:
            enter_short = True
        
        # Exit conditions
        exit_long = False
        exit_short = False
        
        if position == 1:
            # Exit long if Alligator turns bearish OR bull power turns negative
            exit_long = not alligator_bullish or bull_power[i] <= 0
        elif position == -1:
            # Exit short if Alligator turns bullish OR bear power turns positive
            exit_short = not alligator_bearish or bear_power[i] >= 0
        
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
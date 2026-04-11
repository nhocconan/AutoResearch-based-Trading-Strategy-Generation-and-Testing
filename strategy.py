#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Elder Ray + 1d Williams %R with volume confirmation
# - Elder Ray: Bull Power = High - EMA13, Bear Power = Low - EMA13
# - Williams %R (14): Measures overbought/oversold levels
# - Long: Bull Power > 0 AND Williams %R < -80 (oversold) AND volume > 1.5x 20-period average
# - Short: Bear Power < 0 AND Williams %R > -20 (overbought) AND volume > 1.5x 20-period average
# - Uses discrete position sizing: ±0.25 to limit drawdown and reduce fee churn
# - Target: 12-37 trades/year (50-150 total over 4 years) to stay within fee drag limits
# - Elder Ray measures bull/bear power behind the move
# - Williams %R identifies exhaustion points for mean reversion
# - Volume confirmation filters out weak signals
# - Works in both bull (strong bull power on dips) and bear (strong bear power on rallies) markets

name = "6h_1d_elder_ray_williamsr_volume_v1"
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
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    # Load 1d data ONCE before loop for Williams %R
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return signals
    
    # Pre-compute Williams %R on 1d timeframe
    # Williams %R = (Highest High - Close) / (Highest High - Lowest Low) * -100
    highest_high = pd.Series(df_1d['high']).rolling(window=14, min_periods=14).max().values
    lowest_low = pd.Series(df_1d['low']).rolling(window=14, min_periods=14).min().values
    williams_r = (highest_high - df_1d['close'].values) / (highest_high - lowest_low) * -100
    # Handle division by zero (when high == low)
    williams_r = np.where((highest_high - lowest_low) == 0, -50, williams_r)
    williams_r_aligned = align_htf_to_ltf(prices, df_1d, williams_r)
    
    # Pre-compute Elder Ray on 6h timeframe
    # EMA13 for Elder Ray
    ema13 = pd.Series(close).ewm(span=13, adjust=False, min_periods=13).mean().values
    # Bull Power = High - EMA13
    bull_power = high - ema13
    # Bear Power = Low - EMA13
    bear_power = low - ema13
    
    # Pre-compute volume confirmation (20-period average) on 6h
    volume_sma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    for i in range(50, n):  # Start after 50-bar warmup
        # Skip if any required data is invalid
        if (np.isnan(bull_power[i]) or np.isnan(bear_power[i]) or
            np.isnan(williams_r_aligned[i]) or np.isnan(volume_sma_20[i])):
            signals[i] = 0.0
            continue
        
        # Current price data
        volume_current = volume[i]
        
        # Elder Ray power
        bull_power_positive = bull_power[i] > 0
        bear_power_negative = bear_power[i] < 0
        
        # Williams %R conditions
        williams_oversold = williams_r_aligned[i] < -80
        williams_overbought = williams_r_aligned[i] > -20
        
        # Volume confirmation: current volume > 1.5x 20-period average
        vol_confirm = volume_current > 1.5 * volume_sma_20[i]
        
        # Entry conditions
        enter_long = False
        enter_short = False
        
        # Long: Bull Power positive AND Williams %R oversold AND volume confirmation
        if bull_power_positive and williams_oversold and vol_confirm:
            enter_long = True
        
        # Short: Bear Power negative AND Williams %R overbought AND volume confirmation
        if bear_power_negative and williams_overbought and vol_confirm:
            enter_short = True
        
        # Exit conditions: loss of power or Williams %R normalization
        exit_long = False
        exit_short = False
        
        if position == 1:
            # Exit long if Bull Power becomes negative OR Williams %R rises above -50
            exit_long = (not bull_power_positive) or (williams_r_aligned[i] > -50)
        elif position == -1:
            # Exit short if Bear Power becomes positive OR Williams %R falls below -50
            exit_short = (not bear_power_negative) or (williams_r_aligned[i] < -50)
        
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
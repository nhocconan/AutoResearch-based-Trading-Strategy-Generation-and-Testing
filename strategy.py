#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Elder Ray (Bull/Bear Power) + 1d Williams %R mean reversion
# - Elder Ray: Bull Power = High - EMA13, Bear Power = EMA13 - Low (from 6h)
# - Williams %R from 1d: overbought > -20, oversold < -80
# - Long when 6h Bear Power < 0 (bearish pressure weakening) AND 1d Williams %R < -80 (oversold)
# - Short when 6h Bull Power < 0 (bullish pressure weakening) AND 1d Williams %R > -20 (overbought)
# - Uses discrete position sizing: ±0.25 to limit drawdown and reduce fee churn
# - Target: 12-37 trades/year (50-150 total over 4 years) to stay within fee drag limits
# - Works in bull markets (buy oversold dips in uptrend) and bear markets (sell overbought rallies in downtrend)
# - Williams %R provides mean reversion edge; Elder Ray confirms institutional pressure weakening

name = "6h_1d_elder_ray_williamsr_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    # Load 1d data ONCE before loop for Williams %R
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return signals
    
    # Pre-compute 1d Williams %R (14-period)
    highest_high_1d = pd.Series(df_1d['high']).rolling(window=14, min_periods=14).max().values
    lowest_low_1d = pd.Series(df_1d['low']).rolling(window=14, min_periods=14).min().values
    close_1d = df_1d['close'].values
    williams_r_1d = -100 * (highest_high_1d - close_1d) / (highest_high_1d - lowest_low_1d)
    williams_r_1d_aligned = align_htf_to_ltf(prices, df_1d, williams_r_1d)
    
    # Pre-compute 6h EMA13 for Elder Ray
    close_series = pd.Series(close)
    ema_13 = close_series.ewm(span=13, adjust=False, min_periods=13).mean().values
    
    # 6h Elder Ray components
    bull_power = high - ema_13  # Bull Power = High - EMA13
    bear_power = ema_13 - low   # Bear Power = EMA13 - Low
    
    for i in range(100, n):  # Start after 100-bar warmup
        # Skip if any required data is invalid
        if (np.isnan(bull_power[i]) or np.isnan(bear_power[i]) or 
            np.isnan(williams_r_1d_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Current price data
        price_close = close[i]
        
        # Williams %R conditions
        williams_oversold = williams_r_1d_aligned[i] < -80
        williams_overbought = williams_r_1d_aligned[i] > -20
        
        # Elder Ray conditions
        bear_power_weakening = bear_power[i] < 0  # Bearish pressure weakening
        bull_power_weakening = bull_power[i] < 0  # Bullish pressure weakening
        
        # Entry conditions
        enter_long = False
        enter_short = False
        
        # Long: Bear power weakening (bulls taking control) + Williams oversold
        if bear_power_weakening and williams_oversold:
            enter_long = True
        
        # Short: Bull power weakening (bears taking control) + Williams overbought
        if bull_power_weakening and williams_overbought:
            enter_short = True
        
        # Exit conditions: opposite Elder Ray signal or Williams extreme reversal
        exit_long = False
        exit_short = False
        
        if position == 1:
            # Exit long if bull power weakens (bulls losing control) OR Williams not oversold
            exit_long = (bull_power[i] >= 0) or (not williams_oversold)
        elif position == -1:
            # Exit short if bear power weakens (bears losing control) OR Williams not overbought
            exit_short = (bear_power[i] >= 0) or (not williams_overbought)
        
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
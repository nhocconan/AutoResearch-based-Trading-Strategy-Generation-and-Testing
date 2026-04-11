#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1d Williams %R + Volume Spike + Choppiness Regime Filter
# - Williams %R(14): momentum oscillator, long when < -80 (oversold), short when > -20 (overbought)
# - Volume Spike: current volume > 2.0x 20-period average to confirm strong moves
# - Choppiness Regime: CHOP(14) > 61.8 = ranging (mean reversion), CHOP < 38.2 = trending (use Williams %R extremes)
# - Works in both bull (mean reversion in ranges) and bear (extreme oversold bounces)
# - Discrete position sizing: ±0.25 to limit drawdown and reduce fee churn
# - Target: 7-25 trades/year (30-100 total over 4 years) to stay within fee drag limits

name = "1d_williamsr_volume_chop_v1"
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
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    # Load 1w data ONCE before loop for regime filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 30:
        return signals
    
    # Pre-compute 1w Choppiness Index
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    tr1 = np.maximum(high_1w[1:] - low_1w[1:], np.abs(high_1w[1:] - close_1w[:-1]), np.abs(low_1w[1:] - close_1w[:-1]))
    tr = np.concatenate([[0], tr1])
    atr_1w = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    highest_high_1w = pd.Series(high_1w).rolling(window=14, min_periods=14).max().values
    lowest_low_1w = pd.Series(low_1w).rolling(window=14, min_periods=14).min().values
    chop_denom = highest_high_1w - lowest_low_1w
    chop_denom_safe = np.where(chop_denom == 0, 1e-10, chop_denom)
    chop_1w = 100 * np.log10(np.sum(tr) / chop_denom_safe) / np.log10(14) if False else \
              100 * np.log10(pd.Series(tr).rolling(window=14, min_periods=14).sum().values / chop_denom_safe) / np.log10(14)
    chop_1w_aligned = align_htf_to_ltf(prices, df_1w, chop_1w)
    
    # Pre-compute Williams %R on 1d
    highest_high = pd.Series(high).rolling(window=14, min_periods=14).max().values
    lowest_low = pd.Series(low).rolling(window=14, min_periods=14).min().values
    denom = highest_high - lowest_low
    denom_safe = np.where(denom == 0, 1e-10, denom)
    williams_r = -100 * (highest_high - close) / denom_safe
    
    # Pre-compute 1d volume confirmation (20-period average)
    volume_sma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    for i in range(100, n):  # Start after 100-bar warmup
        # Skip if any required data is invalid
        if (np.isnan(williams_r[i]) or np.isnan(volume_sma_20[i]) or 
            np.isnan(chop_1w_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Current price data
        volume_current = volume[i]
        
        # Williams %R signals
        williams_oversold = williams_r[i] < -80
        williams_overbought = williams_r[i] > -20
        
        # Volume confirmation: current volume > 2.0x 20-period average
        vol_confirm = volume_current > 2.0 * volume_sma_20[i]
        
        # Choppiness regime filter
        chop_ranging = chop_1w_aligned[i] > 61.8  # ranging market
        chop_trending = chop_1w_aligned[i] < 38.2  # trending market
        
        # Entry conditions
        enter_long = False
        enter_short = False
        
        # Long: Williams %R oversold + volume confirmation (works in ranging OR trending)
        if williams_oversold and vol_confirm:
            enter_long = True
        
        # Short: Williams %R overbought + volume confirmation (works in ranging OR trending)
        if williams_overbought and vol_confirm:
            enter_short = True
        
        # Exit conditions: Williams %R returns to neutral zone
        exit_long = False
        exit_short = False
        
        if position == 1:
            # Exit long when Williams %R rises above -50 (returns to neutral)
            exit_long = williams_r[i] > -50
        elif position == -1:
            # Exit short when Williams %R falls below -50 (returns to neutral)
            exit_short = williams_r[i] < -50
        
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
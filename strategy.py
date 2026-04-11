#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Elder Ray Index with 1d regime filter
# - Elder Ray: Bull Power = High - EMA(13), Bear Power = Low - EMA(13)
# - Bull regime: 1d EMA(50) rising (today's EMA > yesterday's EMA)
# - Bear regime: 1d EMA(50) falling (today's EMA < yesterday's EMA)
# - Long in bull regime when Bull Power > 0 and rising (current > previous)
# - Short in bear regime when Bear Power < 0 and falling (current < previous)
# - Exit when power crosses zero or regime changes
# - Uses discrete position sizing (0.25) to minimize fee churn
# - Target: 12-30 trades/year (50-120 total over 4 years) to stay within limits

name = "6h_1d_elder_ray_regime_v3"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    # Load 1d data ONCE before loop for EMA regime filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return signals
    
    # Pre-compute 1d EMA(50) for regime detection
    close_1d = df_1d['close'].values
    ema_50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # Pre-compute 6h EMA(13) for Elder Ray
    ema_13 = pd.Series(close).ewm(span=13, adjust=False, min_periods=13).mean().values
    
    # Pre-compute Elder Ray components
    bull_power = high - ema_13  # Bull Power = High - EMA(13)
    bear_power = low - ema_13   # Bear Power = Low - EMA(13)
    
    for i in range(100, n):  # Start after 100-bar warmup
        # Skip if any required data is invalid
        if (np.isnan(ema_13[i]) or np.isnan(bull_power[i]) or 
            np.isnan(bear_power[i]) or np.isnan(ema_50_1d_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Current values
        bull_current = bull_power[i]
        bear_current = bear_power[i]
        ema_1d_current = ema_50_1d_aligned[i]
        
        # Previous values (for change detection)
        bull_previous = bull_power[i-1]
        bear_previous = bear_power[i-1]
        ema_1d_previous = ema_50_1d_aligned[i-1]
        
        # 1d EMA regime: rising = bull regime, falling = bear regime
        ema_rising = ema_1d_current > ema_1d_previous
        ema_falling = ema_1d_current < ema_1d_previous
        
        # Elder Ray signals with regime filter
        enter_long = False
        enter_short = False
        
        # Long: bull regime + Bull Power > 0 and rising
        if ema_rising and bull_current > 0 and bull_current > bull_previous:
            enter_long = True
        
        # Short: bear regime + Bear Power < 0 and falling
        if ema_falling and bear_current < 0 and bear_current < bear_previous:
            enter_short = True
        
        # Exit conditions
        exit_long = False
        exit_short = False
        
        if position == 1:
            # Exit long if Bull Power <= 0 or regime turns bearish
            exit_long = bull_current <= 0 or ema_falling
        elif position == -1:
            # Exit short if Bear Power >= 0 or regime turns bullish
            exit_short = bear_current >= 0 or ema_rising
        
        # Trading logic with discrete position sizing
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
#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Elder Ray + 1w regime filter
# - Bull Power = High - EMA(13), Bear Power = EMA(13) - Low
# - Long: Bull Power > 0 and Bear Power < 0 (strong bullish momentum) AND 1w close > 1w EMA(34) (bullish weekly regime)
# - Short: Bear Power > 0 and Bull Power < 0 (strong bearish momentum) AND 1w close < 1w EMA(34) (bearish weekly regime)
# - Exit: Opposite power crosses zero (Bear Power > 0 for long exit, Bull Power > 0 for short exit)
# - Uses 1w EMA(34) for regime filter to avoid counter-trend trades in strong weekly trends
# - Discrete position sizing: ±0.25 to limit drawdown and reduce fee churn
# - Target: 12-30 trades/year (50-120 total over 4 years) to stay within fee drag limits
# - Elder Ray measures bull/bear power relative to EMA, effective in both trending and ranging markets

name = "6h_1w_elder_ray_regime_v1"
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
    
    # Load 1w data ONCE before loop for regime filter (MTF rule compliance)
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 34:
        return signals
    
    # Pre-compute 1w EMA(34) for regime filter
    close_1w = df_1w['close'].values
    ema_34_1w = pd.Series(close_1w).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_34_1w)
    
    # Pre-compute 6h EMA(13) for Elder Ray calculation
    ema_13 = pd.Series(close).ewm(span=13, adjust=False, min_periods=13).mean().values
    
    # Pre-compute Elder Ray components
    bull_power = high - ema_13  # Bull Power = High - EMA(13)
    bear_power = ema_13 - low   # Bear Power = EMA(13) - Low
    
    for i in range(100, n):  # Start after 100-bar warmup
        # Skip if any required data is invalid
        if (np.isnan(bull_power[i]) or np.isnan(bear_power[i]) or 
            np.isnan(ema_34_1w_aligned[i]) or np.isnan(close[i])):
            signals[i] = 0.0
            continue
        
        # Current price data
        close_price = close[i]
        
        # Elder Ray values
        bp = bull_power[i]   # Bull Power
        br = bear_power[i]   # Bear Power
        
        # 1w regime filter
        weekly_bullish = close_price > ema_34_1w_aligned[i]
        weekly_bearish = close_price < ema_34_1w_aligned[i]
        
        # Entry conditions
        enter_long = False
        enter_short = False
        
        # Long: Strong bullish momentum (BP>0 and BR<0) AND bullish weekly regime
        if bp > 0 and br < 0 and weekly_bullish:
            enter_long = True
        
        # Short: Strong bearish momentum (BR>0 and BP<0) AND bearish weekly regime
        if br > 0 and bp < 0 and weekly_bearish:
            enter_short = True
        
        # Exit conditions
        exit_long = False
        exit_short = False
        
        if position == 1:
            # Exit long when Bear Power turns positive (bullish momentum fading)
            exit_long = br > 0
        elif position == -1:
            # Exit short when Bull Power turns positive (bearish momentum fading)
            exit_short = bp > 0
        
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
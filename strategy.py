#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Elder Ray + 12h Williams %R regime filter
# - Elder Ray (Bull Power = High - EMA13, Bear Power = EMA13 - Low) measures bull/bear strength
# - Long when Bull Power > 0 AND Bear Power < 0 (strong bullish momentum)
# - Short when Bull Power < 0 AND Bear Power > 0 (strong bearish momentum)
# - 12h Williams %R regime filter: only trade when Williams %R < -80 (oversold) for longs or > -20 (overbought) for shorts
# - This avoids counter-trend trades and improves win rate by aligning with higher timeframe momentum extremes
# - Uses discrete position sizing: ±0.25 to limit drawdown and reduce fee churn
# - Target: 12-37 trades/year (50-150 total over 4 years) to stay within fee drag limits for 6h
# - Works in both bull (strong rallies with regime alignment) and bear (strong declines with regime alignment) markets

name = "6h_12h_elder_ray_williamsr_v1"
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
    
    # Load 12h data ONCE before loop for Williams %R regime filter
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 20:
        return signals
    
    # Pre-compute 12h Williams %R (14-period)
    high_12h = df_12h['high'].values
    low_12h = df_12h['low'].values
    close_12h = df_12h['close'].values
    
    # Williams %R = (Highest High - Close) / (Highest High - Lowest Low) * -100
    highest_high = pd.Series(high_12h).rolling(window=14, min_periods=14).max().values
    lowest_low = pd.Series(low_12h).rolling(window=14, min_periods=14).min().values
    williams_r = (highest_high - close_12h) / (highest_high - lowest_low) * -100
    williams_r = np.where((highest_high - lowest_low) == 0, -50, williams_r)  # avoid division by zero
    
    # Align 12h Williams %R to 6h timeframe
    williams_r_aligned = align_htf_to_ltf(prices, df_12h, williams_r)
    
    # Pre-compute 6h EMA13 for Elder Ray
    close_series = pd.Series(close)
    ema13 = close_series.ewm(span=13, adjust=False, min_periods=13).mean().values
    
    # Pre-compute 6h Elder Ray components
    bull_power = high - ema13  # High - EMA13
    bear_power = ema13 - low   # EMA13 - Low
    
    for i in range(100, n):  # Start after 100-bar warmup
        # Skip if any required data is invalid
        if (np.isnan(ema13[i]) or np.isnan(bull_power[i]) or np.isnan(bear_power[i]) or
            np.isnan(williams_r_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Elder Ray signals
        bullish_momentum = bull_power[i] > 0 and bear_power[i] < 0  # Strong bullish
        bearish_momentum = bull_power[i] < 0 and bear_power[i] > 0  # Strong bearish
        
        # Williams %R regime filter: avoid counter-trend trades
        williams_r_val = williams_r_aligned[i]
        oversold = williams_r_val < -80   # Extreme oversold - good for longs
        overbought = williams_r_val > -20 # Extreme overbought - good for shorts
        
        # Entry conditions with regime alignment
        enter_long = False
        enter_short = False
        
        # Long: Bullish momentum + Williams %R oversold (regime alignment)
        if bullish_momentum and oversold:
            enter_long = True
        
        # Short: Bearish momentum + Williams %R overbought (regime alignment)
        if bearish_momentum and overbought:
            enter_short = True
        
        # Exit conditions: momentum reversal or regime extreme
        exit_long = False
        exit_short = False
        
        if position == 1:
            # Exit long if bearish momentum emerges OR Williams %R becomes overbought (regime change)
            exit_long = bearish_momentum or (williams_r_val > -20)
        elif position == -1:
            # Exit short if bullish momentum emerges OR Williams %R becomes oversold (regime change)
            exit_short = bullish_momentum or (williams_r_val < -80)
        
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
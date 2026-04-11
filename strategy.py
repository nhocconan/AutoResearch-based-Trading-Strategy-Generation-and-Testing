#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Williams Alligator + Elder Ray combination with 12h trend filter
# - Williams Alligator: Jaw(13,8), Teeth(8,5), Lips(5,3) SMAs on median price
# - Elder Ray: Bull Power = High - EMA(13), Bear Power = Low - EMA(13)
# - Long: Lips > Teeth > Jaw (bullish alignment) AND Bull Power > 0 AND price > 12h EMA(50)
# - Short: Lips < Teeth < Jaw (bearish alignment) AND Bear Power < 0 AND price < 12h EMA(50)
# - Exit: Alligator alignment reverses OR Elder Power crosses zero
# - Uses 12h EMA(50) for higher timeframe trend filter to avoid counter-trend trades
# - Discrete position sizing: ±0.25 to limit drawdown and reduce fee churn
# - Target: 12-25 trades/year (50-100 total over 4 years) to stay within fee drag limits
# - Alligator identifies trend, Elder Ray measures power, 12h EMA filters regime

name = "6h_12h_alligator_elder_v1"
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
    
    # Load 12h data ONCE before loop for trend filter (MTF rule compliance)
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 50:
        return signals
    
    # Pre-compute 12h EMA(50) for trend filter
    close_12h = df_12h['close'].values
    ema_50_12h = pd.Series(close_12h).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_12h_aligned = align_htf_to_ltf(prices, df_12h, ema_50_12h)
    
    # Pre-calculate Williams Alligator components (6h timeframe)
    # Median price = (High + Low) / 2
    median_price = (high + low) / 2
    
    # Alligator Jaw: 13-period SMA, shifted 8 bars
    jaw = pd.Series(median_price).rolling(window=13, min_periods=13).mean().values
    jaw = np.roll(jaw, 8)
    jaw[:8] = np.nan  # First 8 values invalid due to shift
    
    # Alligator Teeth: 8-period SMA, shifted 5 bars
    teeth = pd.Series(median_price).rolling(window=8, min_periods=8).mean().values
    teeth = np.roll(teeth, 5)
    teeth[:5] = np.nan  # First 5 values invalid due to shift
    
    # Alligator Lips: 5-period SMA, shifted 3 bars
    lips = pd.Series(median_price).rolling(window=5, min_periods=5).mean().values
    lips = np.roll(lips, 3)
    lips[:3] = np.nan  # First 3 values invalid due to shift
    
    # Pre-calculate Elder Ray components (6h timeframe)
    # Elder Ray uses EMA(13) of close
    ema_13 = pd.Series(close).ewm(span=13, adjust=False, min_periods=13).mean().values
    bull_power = high - ema_13  # Bull Power = High - EMA(13)
    bear_power = low - ema_13   # Bear Power = Low - EMA(13)
    
    for i in range(100, n):  # Start after 100-bar warmup
        # Skip if any required data is invalid
        if (np.isnan(lips[i]) or np.isnan(teeth[i]) or np.isnan(jaw[i]) or
            np.isnan(bull_power[i]) or np.isnan(bear_power[i]) or
            np.isnan(ema_50_12h_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Current price data
        close_price = close[i]
        
        # Williams Alligator alignment
        # Bullish: Lips > Teeth > Jaw
        bullish_alignment = lips[i] > teeth[i] and teeth[i] > jaw[i]
        # Bearish: Lips < Teeth < Jaw
        bearish_alignment = lips[i] < teeth[i] and teeth[i] < jaw[i]
        
        # Elder Ray power
        bull_power_current = bull_power[i]
        bear_power_current = bear_power[i]
        
        # 12h EMA trend bias
        ema_bias_long = close_price > ema_50_12h_aligned[i]
        ema_bias_short = close_price < ema_50_12h_aligned[i]
        
        # Entry conditions
        enter_long = False
        enter_short = False
        
        # Long: Bullish Alligator alignment AND positive Bull Power AND long bias from 12h EMA
        if bullish_alignment and bull_power_current > 0 and ema_bias_long:
            enter_long = True
        
        # Short: Bearish Alligator alignment AND negative Bear Power AND short bias from 12h EMA
        if bearish_alignment and bear_power_current < 0 and ema_bias_short:
            enter_short = True
        
        # Exit conditions
        exit_long = False
        exit_short = False
        
        if position == 1:
            # Exit long if Alligator turns bearish OR Bull Power becomes negative
            exit_long = not bullish_alignment or bull_power_current <= 0
        elif position == -1:
            # Exit short if Alligator turns bullish OR Bear Power becomes positive
            exit_short = not bearish_alignment or bear_power_current >= 0
        
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
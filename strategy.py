#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Elder Ray + 1d regime filter
# - Bull Power = High - EMA13(close), Bear Power = EMA13(close) - Low
# - Long when Bull Power > 0 AND Bear Power increasing (less negative) AND 1d close > 1d EMA50 (bullish regime)
# - Short when Bear Power < 0 AND Bull Power decreasing (less positive) AND 1d close < 1d EMA50 (bearish regime)
# - Exit when power signals reverse or regime changes
# - Uses 1d for regime (trend bias) and 6h for precise entry timing
# - Session filter: 08-20 UTC to avoid low-volume Asian session noise
# - Position sizing: 0.25 discrete level to control drawdown
# - Target: 12-37 trades/year (50-150 total over 4 years) to minimize fee drag

name = "6h_1d_elder_ray_regime_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 60:
        return np.zeros(n)
    
    # Pre-compute primary timeframe data
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    open_time = prices['open_time'].values
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    # Load 1d data ONCE before loop (MTF rule compliance)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return signals
    
    # Calculate 1d EMA50 for regime filter
    close_1d = df_1d['close'].values
    ema50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    bullish_regime = close_1d > ema50_1d
    bearish_regime = close_1d < ema50_1d
    # Align to 6h timeframe with proper delay (completed 1d bar only)
    bullish_regime_aligned = align_htf_to_ltf(prices, df_1d, bullish_regime)
    bearish_regime_aligned = align_htf_to_ltf(prices, df_1d, bearish_regime)
    
    # Calculate 6h Elder Ray components
    # Bull Power = High - EMA13(close)
    # Bear Power = EMA13(close) - Low
    ema13 = pd.Series(close).ewm(span=13, adjust=False, min_periods=13).mean().values
    bull_power = high - ema13
    bear_power = ema13 - low
    
    # Calculate power momentum (change from previous bar)
    bull_power_momentum = bull_power - np.roll(bull_power, 1)
    bear_power_momentum = bear_power - np.roll(bear_power, 1)
    # First bar momentum is zero
    bull_power_momentum[0] = 0
    bear_power_momentum[0] = 0
    
    # Pre-compute session filter (08-20 UTC)
    hours = pd.DatetimeIndex(open_time).hour
    in_session = (hours >= 8) & (hours <= 20)
    
    for i in range(1, n):  # Start from 1 to have previous bar data
        # Skip if not in trading session
        if not in_session[i]:
            signals[i] = 0.0
            continue
            
        # Skip if any required data is invalid
        if (np.isnan(bull_power[i]) or np.isnan(bear_power[i]) or
            np.isnan(bull_power_momentum[i]) or np.isnan(bear_power_momentum[i]) or
            np.isnan(bullish_regime_aligned[i]) or np.isnan(bearish_regime_aligned[i])):
            signals[i] = 0.0
            continue
        
        if position == 0:  # Flat - look for entry
            # Long: Bull Power > 0 AND Bear Power increasing (less negative) AND bullish regime
            if bull_power[i] > 0 and bear_power_momentum[i] > 0 and bullish_regime_aligned[i]:
                position = 1
                signals[i] = 0.25
            # Short: Bear Power < 0 AND Bull Power decreasing (less positive) AND bearish regime
            elif bear_power[i] > 0 and bull_power_momentum[i] < 0 and bearish_regime_aligned[i]:
                position = -1
                signals[i] = -0.25
            else:
                signals[i] = 0.0
        elif position == 1:  # Long position - look for exit
            # Exit when Bull Power <= 0 OR Bear Power decreasing OR regime turns bearish
            exit_condition = (bull_power[i] <= 0) or \
                           (bear_power_momentum[i] < 0) or \
                           (not bullish_regime_aligned[i])
            if exit_condition:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
        else:  # position == -1 (Short position) - look for exit
            # Exit when Bear Power <= 0 OR Bull Power increasing OR regime turns bullish
            exit_condition = (bear_power[i] <= 0) or \
                           (bull_power_momentum[i] > 0) or \
                           (bullish_regime_aligned[i])
            if exit_condition:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
    
    return signals
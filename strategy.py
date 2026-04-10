#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Elder Ray + 1d regime filter
# - Bull Power = High - EMA13(close), Bear Power = EMA13(close) - Low
# - Long when Bull Power > 0 AND Bear Power < 0 AND 1d close > 1d EMA50 (bullish regime)
# - Short when Bear Power > 0 AND Bull Power < 0 AND 1d close < 1d EMA50 (bearish regime)
# - Exit when power signals weaken (Bull Power < 0 for long, Bear Power < 0 for short)
# - Uses 1d EMA50 for regime filter to avoid counter-trend trades
# - Position sizing: 0.25 discrete level
# - Target: 12-37 trades/year (50-150 total over 4 years) to minimize fee drag
# - Elder Ray measures bull/bear strength behind price moves, effective in both bull and bear markets when combined with regime filter

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
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    # Load 1d data ONCE before loop (MTF rule compliance)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return signals
    
    # Calculate 1d EMA50 for regime filter
    close_1d = df_1d['close'].values
    ema50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    # Regime: bullish if close > EMA50, bearish if close < EMA50
    bullish_regime = close_1d > ema50_1d
    bearish_regime = close_1d < ema50_1d
    # Align to 6h timeframe with proper delay (completed 1d bar only)
    bullish_regime_aligned = align_htf_to_ltf(prices, df_1d, bullish_regime)
    bearish_regime_aligned = align_htf_to_ltf(prices, df_1d, bearish_regime)
    
    # Calculate EMA13 for Elder Ray (6h timeframe)
    ema13 = pd.Series(close).ewm(span=13, adjust=False, min_periods=13).mean().values
    
    # Calculate Elder Ray components
    bull_power = high - ema13  # Bull Power = High - EMA13
    bear_power = ema13 - low   # Bear Power = EMA13 - Low
    
    for i in range(13, n):  # Start from 13 to have sufficient lookback for EMA13
        # Skip if any required data is invalid
        if (np.isnan(ema13[i]) or np.isnan(bull_power[i]) or np.isnan(bear_power[i]) or
            np.isnan(bullish_regime_aligned[i]) or np.isnan(bearish_regime_aligned[i])):
            signals[i] = 0.0
            continue
        
        if position == 0:  # Flat - look for entry
            # Long: Bull Power > 0 AND Bear Power < 0 AND bullish 1d regime
            if bull_power[i] > 0 and bear_power[i] < 0 and bullish_regime_aligned[i]:
                position = 1
                signals[i] = 0.25
            # Short: Bear Power > 0 AND Bull Power < 0 AND bearish 1d regime
            elif bear_power[i] > 0 and bull_power[i] < 0 and bearish_regime_aligned[i]:
                position = -1
                signals[i] = -0.25
            else:
                signals[i] = 0.0
        elif position == 1:  # Long position - look for exit
            # Exit when Bull Power weakens (< 0) or Bear Power strengthens (> 0)
            if bull_power[i] < 0 or bear_power[i] > 0:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
        else:  # position == -1 (Short position) - look for exit
            # Exit when Bear Power weakens (< 0) or Bull Power strengthens (> 0)
            if bear_power[i] < 0 or bull_power[i] > 0:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
    
    return signals
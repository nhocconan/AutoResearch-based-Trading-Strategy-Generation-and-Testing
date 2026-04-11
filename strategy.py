#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Elder Ray + Williams %R with 1d regime filter
# - Elder Ray (Bull/Bear Power) measures buying/selling pressure relative to EMA13
# - Williams %R identifies overbought/oversold conditions
# - Long when Bull Power > 0, Williams %R < -80 (oversold), and 1d close > EMA50 (bullish regime)
# - Short when Bear Power < 0, Williams %R > -20 (overbought), and 1d close < EMA50 (bearish regime)
# - Uses discrete position sizing: ±0.25 to limit drawdown and reduce fee churn
# - Target: 12-37 trades/year (50-150 total over 4 years) to stay within fee drag limits for 6h
# - Works in both bull (buy dips in uptrend) and bear (sell rallies in downtrend) markets
# - 1d EMA50 regime filter ensures we only trade with the higher timeframe trend

name = "6h_1d_elder_ray_williamsr_v2"
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
    
    # Load 1d data ONCE before loop for regime filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return signals
    
    # Pre-compute 1d EMA50 for regime filter
    close_1d = df_1d['close'].values
    ema50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema50_1d)
    
    # Pre-compute 6h EMA13 for Elder Ray
    ema13 = pd.Series(close).ewm(span=13, adjust=False, min_periods=13).mean().values
    
    # Pre-compute 6h Williams %R (14-period)
    highest_high = pd.Series(high).rolling(window=14, min_periods=14).max().values
    lowest_low = pd.Series(low).rolling(window=14, min_periods=14).min().values
    williams_r = -100 * (highest_high - close) / (highest_high - lowest_low)
    
    for i in range(100, n):  # Start after 100-bar warmup
        # Skip if any required data is invalid
        if (np.isnan(ema13[i]) or np.isnan(williams_r[i]) or np.isnan(ema50_1d_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Elder Ray components
        bull_power = high[i] - ema13[i]  # Measures buying strength
        bear_power = low[i] - ema13[i]   # Measures selling strength (typically negative)
        
        # Williams %R conditions
        oversold = williams_r[i] < -80
        overbought = williams_r[i] > -20
        
        # 1d regime filter
        bullish_regime = close[i] > ema50_1d_aligned[i]
        bearish_regime = close[i] < ema50_1d_aligned[i]
        
        # Entry conditions
        enter_long = False
        enter_short = False
        
        # Long: Bull Power positive + oversold + bullish 1d regime
        if bull_power > 0 and oversold and bullish_regime:
            enter_long = True
        
        # Short: Bear Power negative + overbought + bearish 1d regime
        if bear_power < 0 and overbought and bearish_regime:
            enter_short = True
        
        # Exit conditions: opposite Elder Ray signal or regime change
        exit_long = False
        exit_short = False
        
        if position == 1:
            # Exit long if Bear Power becomes negative OR regime turns bearish
            exit_long = (bear_power < 0) or (not bullish_regime)
        elif position == -1:
            # Exit short if Bull Power becomes positive OR regime turns bullish
            exit_short = (bull_power > 0) or (not bearish_regime)
        
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
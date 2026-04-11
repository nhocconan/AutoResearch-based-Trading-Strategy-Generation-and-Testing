#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Elder Ray + 1d Williams %R with regime filter
# - Elder Ray (Bull Power = High - EMA13, Bear Power = EMA13 - Low) from 6h measures bull/bear strength
# - Williams %R from 1d identifies overbought/oversold conditions (%R < -80 oversold, > -20 overbought)
# - Long when Bull Power > 0 (bullish momentum) AND Williams %R < -80 (oversold) AND 6h close > 6h EMA50 (uptrend filter)
# - Short when Bear Power > 0 (bearish momentum) AND Williams %R > -20 (overbought) AND 6h close < 6h EMA50 (downtrend filter)
# - Uses discrete position sizing: ±0.25 to limit drawdown and reduce fee churn
# - Target: 12-37 trades/year (50-150 total over 4 years) to stay within fee drag limits for 6h
# - Works in both bull (buy oversold in uptrend) and bear (sell overbought in downtrend) markets
# - Williams %R provides mean reversion edge in ranging markets, Elder Ray provides trend confirmation

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
    williams_r_1d = np.where((highest_high_1d - lowest_low_1d) == 0, -50, williams_r_1d)  # handle division by zero
    
    # Align 1d Williams %R to 6h timeframe
    williams_r_1d_aligned = align_htf_to_ltf(prices, df_1d, williams_r_1d)
    
    # Pre-compute 6h indicators
    # EMA13 for Elder Ray
    close_series = pd.Series(close)
    ema13 = close_series.ewm(span=13, adjust=False, min_periods=13).mean().values
    # EMA50 for trend filter
    ema50 = close_series.ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Elder Ray components
    bull_power = high - ema13  # Bull Power = High - EMA13
    bear_power = ema13 - low   # Bear Power = EMA13 - Low
    
    for i in range(100, n):  # Start after 100-bar warmup
        # Skip if any required data is invalid
        if (np.isnan(ema13[i]) or np.isnan(ema50[i]) or np.isnan(williams_r_1d_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Current price data
        price_close = close[i]
        price_high = high[i]
        price_low = low[i]
        
        # Elder Ray values
        bull_power_current = bull_power[i]
        bear_power_current = bear_power[i]
        
        # Williams %R conditions
        williams_r_current = williams_r_1d_aligned[i]
        williams_oversold = williams_r_current < -80
        williams_overbought = williams_r_current > -20
        
        # Trend filter
        uptrend = price_close > ema50[i]
        downtrend = price_close < ema50[i]
        
        # Entry conditions
        enter_long = False
        enter_short = False
        
        # Long: Bull Power > 0 (bullish momentum) AND Williams %R < -80 (oversold) AND uptrend
        if bull_power_current > 0 and williams_oversold and uptrend:
            enter_long = True
        
        # Short: Bear Power > 0 (bearish momentum) AND Williams %R > -20 (overbought) AND downtrend
        if bear_power_current > 0 and williams_overbought and downtrend:
            enter_short = True
        
        # Exit conditions: opposite Elder Ray signal or Williams %R reversal
        exit_long = False
        exit_short = False
        
        if position == 1:
            # Exit long if Bear Power > 0 (bearish momentum takes over) OR Williams %R > -50 (reverts from oversold)
            exit_long = (bear_power_current > 0) or (williams_r_current > -50)
        elif position == -1:
            # Exit short if Bull Power > 0 (bullish momentum takes over) OR Williams %R < -50 (reverts from overbought)
            exit_short = (bull_power_current > 0) or (williams_r_current < -50)
        
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
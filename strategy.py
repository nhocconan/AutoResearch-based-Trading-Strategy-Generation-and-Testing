#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Williams %R mean reversion with 1d trend filter
# - Williams %R(14) from 6h: identifies overbought/oversold conditions
# - Long when %R < -80 (oversold) and 1d EMA(50) > EMA(200) (bullish trend)
# - Short when %R > -20 (overbought) and 1d EMA(50) < EMA(200) (bearish trend)
# - Uses discrete position sizing: ±0.25 to limit drawdown and reduce fee churn
# - Target: 12-37 trades/year (50-150 total over 4 years) to stay within fee drag limits for 6h
# - Works in both bull (mean reversion in uptrend) and bear (mean reversion in downtrend) markets
# - Williams %R provides clear reversal signals, 1d EMA filter ensures we trade with higher timeframe trend

name = "6h_1d_williamsr_trendfilter_v1"
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
    
    # Load 1d data ONCE before loop for trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 200:
        return signals
    
    # Pre-compute 1d EMAs for trend filter
    close_1d = df_1d['close'].values
    ema_50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_200_1d = pd.Series(close_1d).ewm(span=200, adjust=False, min_periods=200).mean().values
    
    # Align 1d EMAs to 6h timeframe
    ema_50_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    ema_200_aligned = align_htf_to_ltf(prices, df_1d, ema_200_1d)
    
    # Pre-compute 6h Williams %R (14-period)
    highest_high = pd.Series(high).rolling(window=14, min_periods=14).max().values
    lowest_low = pd.Series(low).rolling(window=14, min_periods=14).min().values
    williams_r = -100 * (highest_high - close) / (highest_low - lowest_low + 1e-10)  # Add small epsilon to avoid division by zero
    
    for i in range(100, n):  # Start after 100-bar warmup
        # Skip if any required data is invalid
        if (np.isnan(williams_r[i]) or np.isnan(ema_50_aligned[i]) or np.isnan(ema_200_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Trend filter: bullish if EMA50 > EMA200, bearish if EMA50 < EMA200
        bullish_trend = ema_50_aligned[i] > ema_200_aligned[i]
        bearish_trend = ema_50_aligned[i] < ema_200_aligned[i]
        
        # Williams %R conditions
        oversold = williams_r[i] < -80
        overbought = williams_r[i] > -20
        
        # Entry conditions
        enter_long = False
        enter_short = False
        
        # Long: Oversold + bullish trend
        if oversold and bullish_trend:
            enter_long = True
        
        # Short: Overbought + bearish trend
        if overbought and bearish_trend:
            enter_short = True
        
        # Exit conditions: opposite Williams %R level or trend change
        exit_long = False
        exit_short = False
        
        if position == 1:
            # Exit long if overbought OR trend turns bearish
            exit_long = (williams_r[i] > -20) or (not bullish_trend)
        elif position == -1:
            # Exit short if oversold OR trend turns bullish
            exit_short = (williams_r[i] < -80) or (not bearish_trend)
        
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
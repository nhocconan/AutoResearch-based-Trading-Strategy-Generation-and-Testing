#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Williams %R mean reversion with 1d regime filter
# - Williams %R(14) identifies overbought/oversold conditions on 6h
# - Long when %R < -80 (oversold) and 1d close > 1d EMA50 (bullish regime)
# - Short when %R > -20 (overbought) and 1d close < 1d EMA50 (bearish regime)
# - Uses discrete position sizing: ±0.25 to limit drawdown and reduce fee churn
# - Target: 12-37 trades/year (50-150 total over 4 years) to stay within fee drag limits
# - Mean reversion works in ranging markets, regime filter avoids counter-trend trades in strong trends
# - Williams %R is effective for BTC/ETH mean reversion on 6h timeframe

name = "6h_1d_williamsr_regime_v1"
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
    ema_50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # Pre-compute 6h Williams %R (14-period)
    highest_high = pd.Series(high).rolling(window=14, min_periods=14).max().values
    lowest_low = pd.Series(low).rolling(window=14, min_periods=14).min().values
    williams_r = -100 * (highest_high - close) / (highest_high - lowest_low)
    # Handle division by zero when highest_high == lowest_low
    williams_r = np.where((highest_high - lowest_low) == 0, -50, williams_r)
    
    for i in range(100, n):  # Start after 100-bar warmup
        # Skip if any required data is invalid
        if (np.isnan(williams_r[i]) or np.isnan(ema_50_1d_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Current price data
        price_close = close[i]
        wr = williams_r[i]
        regime = ema_50_1d_aligned[i]
        
        # Entry conditions
        enter_long = False
        enter_short = False
        
        # Long: Williams %R oversold (< -80) AND bullish regime (price > EMA50)
        if wr < -80 and price_close > regime:
            enter_long = True
        
        # Short: Williams %R overbought (> -20) AND bearish regime (price < EMA50)
        if wr > -20 and price_close < regime:
            enter_short = True
        
        # Exit conditions: opposite Williams %R level or regime change
        exit_long = False
        exit_short = False
        
        if position == 1:
            # Exit long if Williams %R becomes overbought OR regime turns bearish
            exit_long = (wr > -20) or (price_close < regime)
        elif position == -1:
            # Exit short if Williams %R becomes oversold OR regime turns bullish
            exit_short = (wr < -80) or (price_close > regime)
        
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
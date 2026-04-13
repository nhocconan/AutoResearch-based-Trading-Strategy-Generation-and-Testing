#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Hypothesis: 6h Williams %R mean reversion with 1w trend filter
    # Long when Williams %R(14) < -80 (oversold) + price > 1w EMA50 (uptrend)
    # Short when Williams %R(14) > -20 (overbought) + price < 1w EMA50 (downtrend)
    # Exit when Williams %R returns to -50 (mean reversion midpoint)
    # Uses discrete position sizing (0.25) to minimize fee churn
    # Williams %R is effective in ranging markets; 1w EMA filter avoids counter-trend trades
    # Target: 50-150 total trades over 4 years (~12-37/year) to avoid fee drag
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    
    # Get 1w data (call ONCE before loop)
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 50:
        return np.zeros(n)
    
    # Calculate 1w EMA50 for trend filter
    close_1w = df_1w['close'].values
    ema_50_1w = pd.Series(close_1w).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_50_1w)
    
    # Calculate Williams %R on 6h data
    # Williams %R = (Highest High - Close) / (Highest High - Lowest Low) * -100
    highest_high = pd.Series(high).rolling(window=14, min_periods=14).max().values
    lowest_low = pd.Series(low).rolling(window=14, min_periods=14).min().values
    williams_r = (highest_high - close) / (highest_high - lowest_low) * -100
    # Handle division by zero (when high == low)
    williams_r = np.where((highest_high - lowest_low) == 0, -50, williams_r)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(100, n):
        # Skip if data not ready
        if (np.isnan(ema_50_1w_aligned[i]) or 
            np.isnan(williams_r[i])):
            signals[i] = 0.0
            continue
        
        # Williams %R conditions
        oversold = williams_r[i] < -80
        overbought = williams_r[i] > -20
        mean_reversion_exit = abs(williams_r[i]) < 50  # Williams %R between -50 and 50
        
        # Trend filter: price relative to 1w EMA50
        price_above_ema = close[i] > ema_50_1w_aligned[i]
        price_below_ema = close[i] < ema_50_1w_aligned[i]
        
        # Entry conditions
        bullish_entry = oversold and price_above_ema
        bearish_entry = overbought and price_below_ema
        
        # Exit conditions
        long_exit = mean_reversion_exit or (position == 1 and not price_above_ema)
        short_exit = mean_reversion_exit or (position == -1 and not price_below_ema)
        
        if bullish_entry and position != 1:
            position = 1
            signals[i] = 0.25
        elif bearish_entry and position != -1:
            position = -1
            signals[i] = -0.25
        elif position == 1 and long_exit:
            position = 0
            signals[i] = 0.0
        elif position == -1 and short_exit:
            position = 0
            signals[i] = 0.0
        else:
            # Hold current position
            if position == 1:
                signals[i] = 0.25
            elif position == -1:
                signals[i] = -0.25
            else:
                signals[i] = 0.0
    
    return signals

name = "6h_1w_williams_r_mean_reversion_v1"
timeframe = "6h"
leverage = 1.0
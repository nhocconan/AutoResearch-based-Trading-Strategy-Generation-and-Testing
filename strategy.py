#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Williams %R mean reversion with 1d/1w trend filter
# - Long when Williams %R(14) < -80 (oversold) AND 1d close > 1w EMA(50) (bullish regime)
# - Short when Williams %R(14) > -20 (overbought) AND 1d close < 1w EMA(50) (bearish regime)
# - Exit when Williams %R crosses -50 (mean reversion midpoint)
# - Uses discrete position sizing 0.25 to limit fee churn
# - Target: 12-37 trades/year on 6h timeframe (50-150 total over 4 years)
# - Williams %R captures extreme momentum exhaustion
# - 1w EMA(50) filter ensures we trade with the higher timeframe trend
# - Mean reversion exit reduces whipsaw in ranging markets

name = "6h_1d_1w_williamsr_regime_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Load HTF data ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    df_1w = get_htf_data(prices, '1w')
    if len(df_1d) < 50 or len(df_1w) < 50:
        return np.zeros(n)
    
    # Pre-compute 6h Williams %R (14)
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    
    highest_high = pd.Series(high).rolling(window=14, min_periods=14).max().values
    lowest_low = pd.Series(low).rolling(window=14, min_periods=14).min().values
    williams_r = -100 * (highest_high - close) / (highest_high - lowest_low)
    
    # Pre-compute 1d close for regime filter
    close_1d = df_1d['close'].values
    
    # Pre-compute 1w EMA(50) for trend filter
    close_1w = df_1w['close'].values
    ema_50_1w = pd.Series(close_1w).ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Align HTF indicators to 6h timeframe
    close_1d_aligned = align_htf_to_ltf(prices, df_1d, close_1d)
    ema_50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_50_1w)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(50, n):  # Start after warmup
        # Skip if any required data is invalid
        if (np.isnan(williams_r[i]) or np.isnan(close_1d_aligned[i]) or 
            np.isnan(ema_50_1w_aligned[i])):
            if position == 0:
                signals[i] = 0.0
            elif position == 1:
                signals[i] = 0.25
            else:
                signals[i] = -0.25
            continue
        
        if position == 0:  # Flat - look for new entries
            # Long conditions: Williams %R oversold AND bullish regime (1d close > 1w EMA50)
            if (williams_r[i] < -80 and 
                close_1d_aligned[i] > ema_50_1w_aligned[i]):
                position = 1
                signals[i] = 0.25
            # Short conditions: Williams %R overbought AND bearish regime (1d close < 1w EMA50)
            elif (williams_r[i] > -20 and 
                  close_1d_aligned[i] < ema_50_1w_aligned[i]):
                position = -1
                signals[i] = -0.25
            else:
                signals[i] = 0.0
        else:  # Have position - look for exit
            # Exit condition: Williams %R crosses -50 (mean reversion midpoint)
            exit_signal = ((position == 1 and williams_r[i] > -50) or 
                          (position == -1 and williams_r[i] < -50))
            
            if exit_signal:
                position = 0
                signals[i] = 0.0
            else:
                if position == 1:
                    signals[i] = 0.25
                else:
                    signals[i] = -0.25
    
    return signals
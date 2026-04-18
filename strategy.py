# 1d_Aggressive10_PostClose_Trend
# Aggressive10 strategy on 1d: enter long/short on open after close crosses above/below Aggressive10, exit on reverse cross.
# Works in bull markets (trend following) and bear markets (short signals). Low trade frequency (~10-25/year) minimizes fee drag.
# Uses 1w EMA50 as trend filter for higher-timeframe context.

#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "1d_Aggressive10_PostClose_Trend"
timeframe = "1d"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    open_price = prices['open'].values
    
    # Get 1w data for EMA50 trend filter
    df_1w = get_htf_data(prices, '1w')
    
    # Calculate EMA50 on 1w data
    close_1w = pd.Series(df_1w['close'].values)
    ema_50_1w = close_1w.ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Align EMA50 to daily timeframe
    ema_50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_50_1w)
    
    # Calculate Aggressive10 (10-period EMA of close)
    close_series = pd.Series(close)
    aggressive10 = close_series.ewm(span=10, adjust=False, min_periods=10).values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 50  # Wait for indicator calculations
    
    for i in range(start_idx, n):
        # Skip if any required data is not available
        if np.isnan(aggressive10[i]) or np.isnan(ema_50_1w_aligned[i]):
            signals[i] = 0.0
            continue
        
        open_val = open_price[i]
        prev_close = close[i-1]
        prev_aggressive10 = aggressive10[i-1]
        prev_ema = ema_50_1w_aligned[i-1]
        curr_aggressive10 = aggressive10[i]
        curr_ema = ema_50_1w_aligned[i]
        
        if position == 0:
            # Long: Previous close below Aggressive10, current open above Aggressive10, and price above weekly EMA50
            if prev_close <= prev_aggressive10 and open_val > curr_aggressive10 and open_val > curr_ema:
                signals[i] = 0.30
                position = 1
            # Short: Previous close above Aggressive10, current open below Aggressive10, and price below weekly EMA50
            elif prev_close >= prev_aggressive10 and open_val < curr_aggressive10 and open_val < curr_ema:
                signals[i] = -0.30
                position = -1
        
        elif position == 1:
            # Long exit: Current open below Aggressive10
            if open_val < curr_aggressive10:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.30
        
        elif position == -1:
            # Short exit: Current open above Aggressive10
            if open_val > curr_aggressive10:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.30
    
    return signals
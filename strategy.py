#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Williams %R with 1w trend filter and volume confirmation
# Long when Williams %R crosses above -80 from below (oversold bounce) with volume > 1.5x 24-bar average and price > 1w EMA50 (uptrend)
# Short when Williams %R crosses below -20 from above (overbought rejection) with volume > 1.5x 24-bar average and price < 1w EMA50 (downtrend)
# Exit when Williams %R crosses opposite threshold (-20 for long, -80 for short) or trend fails
# Williams %R captures momentum reversals; weekly EMA50 filters for primary trend; volume confirms conviction.
# Target: 80-180 total trades over 4 years = 20-45/year. Uses discrete sizing (0.25) to minimize fee churn.

name = "6h_WilliamsR_1wEMA50_Volume_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Load 1w data ONCE before loop for trend filter
    df_1w = get_htf_data(prices, '1w')
    
    # Calculate 1w EMA50 for trend filter
    close_1w = df_1w['close'].values
    ema_50_1w = pd.Series(close_1w).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_50_1w)
    
    # Calculate Williams %R (14-period) on 6h data
    highest_high = pd.Series(high).rolling(window=14, min_periods=14).max().values
    lowest_low = pd.Series(low).rolling(window=14, min_periods=14).min().values
    williams_r = -100 * (highest_high - close) / (highest_high - lowest_low)
    # Handle division by zero (when high == low)
    williams_r = np.where((highest_high - lowest_low) == 0, -50, williams_r)
    
    # Volume confirmation (1.5x 24-period average on 6h)
    vol_ma = pd.Series(volume).rolling(window=24, min_periods=24).mean().shift(1).values
    volume_spike = volume > (vol_ma * 1.5)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after warmup (need enough for all calculations)
    start_idx = max(50, 14, 24) + 1  # EMA50(1w) + Williams %R(14) + volume MA(24) + shift(1)
    
    for i in range(start_idx, n):
        # Check for NaN values in indicators
        if (np.isnan(ema_50_1w_aligned[i]) or np.isnan(williams_r[i]) or 
            np.isnan(volume_spike[i])):
            signals[i] = 0.0
            continue
        
        if position == 0:  # Flat - look for new entries
            # Long entry: Williams %R crosses above -80 from below with volume spike and price > 1w EMA50 (uptrend)
            if (williams_r[i] > -80 and williams_r[i-1] <= -80 and 
                volume_spike[i] and close[i] > ema_50_1w_aligned[i]):
                signals[i] = 0.25
                position = 1
            # Short entry: Williams %R crosses below -20 from above with volume spike and price < 1w EMA50 (downtrend)
            elif (williams_r[i] < -20 and williams_r[i-1] >= -20 and 
                  volume_spike[i] and close[i] < ema_50_1w_aligned[i]):
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        
        elif position == 1:  # Long position
            # Exit: Williams %R crosses below -20 (overbought) or price < 1w EMA50 (trend failure)
            if (williams_r[i] < -20 and williams_r[i-1] >= -20) or \
               (close[i] < ema_50_1w_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:  # Short position
            # Exit: Williams %R crosses above -80 (oversold) or price > 1w EMA50 (trend failure)
            if (williams_r[i] > -80 and williams_r[i-1] <= -80) or \
               (close[i] > ema_50_1w_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals
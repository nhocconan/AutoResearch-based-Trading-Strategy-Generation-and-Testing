# 1d_Bankroll_Growth_Compound_v1
# Strategy: 1-day Williams %R mean reversion with volume spike and weekly trend filter
# Williams %R < -80 = oversold (long), > -20 = overbought (short)
# Volume > 1.5x 20-day average confirms momentum
# Weekly EMA50 filter ensures alignment with higher timeframe trend
# Designed for 1-5 trades per week, targeting 30-100 trades over 4 years
# Works in bull/bear: mean reversion in ranges, trend filter avoids counter-trend in strong moves

#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "1d_Bankroll_Growth_Compound_v1"
timeframe = "1d"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get weekly data for trend filter
    df_1w = get_htf_data(prices, '1w')
    
    # Williams %R (14-period)
    highest_high = pd.Series(high).rolling(window=14, min_periods=14).max().values
    lowest_low = pd.Series(low).rolling(window=14, min_periods=14).min().values
    williams_r = -100 * (highest_high - close) / (highest_high - lowest_low)
    # Handle division by zero when high == low
    williams_r = np.where((highest_high - lowest_low) == 0, -50, williams_r)
    
    # Volume spike: volume > 1.5x 20-day average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (1.5 * vol_ma)
    
    # Weekly EMA50 for trend filter
    close_1w = df_1w['close'].values
    ema50_1w = pd.Series(close_1w).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema50_1w)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 50
    
    for i in range(start_idx, n):
        if np.isnan(williams_r[i]) or np.isnan(vol_ma[i]) or np.isnan(ema50_1w_aligned[i]):
            signals[i] = 0.0
            continue
        
        # Entry conditions
        oversold = williams_r[i] < -80
        overbought = williams_r[i] > -20
        vol_spike = volume_spike[i]
        
        # Trend filter: price above/below weekly EMA50
        above_weekly_ema = close[i] > ema50_1w_aligned[i]
        below_weekly_ema = close[i] < ema50_1w_aligned[i]
        
        if position == 0:
            # Long: oversold + volume spike + above weekly EMA
            if oversold and vol_spike and above_weekly_ema:
                signals[i] = 0.25
                position = 1
            # Short: overbought + volume spike + below weekly EMA
            elif overbought and vol_spike and below_weekly_ema:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit: Williams %R returns to neutral or reverses
            if williams_r[i] > -50:  # Exit when momentum fades
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit: Williams %R returns to neutral or reverses
            if williams_r[i] < -50:  # Exit when momentum fades
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals
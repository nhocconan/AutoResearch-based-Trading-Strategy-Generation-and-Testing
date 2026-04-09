#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1h trend-following strategy using 4h/1d Supertrend with volume confirmation
# - 4h/1d Supertrend (ATR=10, mult=3.0) identifies primary trend direction
# - 1h RSI(14) filter: only long when RSI > 50 in uptrend, short when RSI < 50 in downtrend
# - Volume confirmation: current volume > 1.3x 20-period average to avoid false breakouts
# - Session filter (08-20 UTC) to focus on high-liquidity hours
# - Fixed position size 0.20 to control drawdown and minimize fee churn
# - Target: 15-30 trades/year on 1h timeframe (60-120 total over 4 years)
# - Works in bull markets (follow uptrend) and bear markets (follow downtrend)

name = "1h_4h_1d_supertrend_rsi_volume_v1"
timeframe = "1h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    open_time = prices['open_time'].values
    
    # Load 4h and 1d data ONCE before loop
    df_4h = get_htf_data(prices, '4h')
    df_1d = get_htf_data(prices, '1d')
    if len(df_4h) < 30 or len(df_1d) < 30:
        return np.zeros(n)
    
    high_4h = df_4h['high'].values
    low_4h = df_4h['low'].values
    close_4h = df_4h['close'].values
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate 4h Supertrend (ATR=10, mult=3.0)
    atr_period = 10
    multiplier = 3.0
    
    # True Range for 4h
    tr1_4h = high_4h - low_4h
    tr2_4h = np.abs(high_4h - np.roll(close_4h, 1))
    tr3_4h = np.abs(low_4h - np.roll(close_4h, 1))
    tr_4h = np.maximum(tr1_4h, np.maximum(tr2_4h, tr3_4h))
    tr_4h[0] = tr1_4h[0]  # First value
    atr_4h = pd.Series(tr_4h).rolling(window=atr_period, min_periods=atr_period).mean().values
    
    # Basic Upper and Lower Bands
    basic_ub_4h = (high_4h + low_4h) / 2 + multiplier * atr_4h
    basic_lb_4h = (high_4h + low_4h) / 2 - multiplier * atr_4h
    
    # Initialize Supertrend
    supertrend_4h = np.full_like(close_4h, np.nan, dtype=float)
    direction_4h = np.full_like(close_4h, 1, dtype=float)  # 1 for uptrend, -1 for downtrend
    
    for i in range(1, len(close_4h)):
        if i < atr_period:
            continue
            
        # Final Upper Band
        if basic_ub_4h[i] < supertrend_4h[i-1] or close_4h[i-1] > supertrend_4h[i-1]:
            final_ub = basic_ub_4h[i]
        else:
            final_ub = supertrend_4h[i-1]
            
        # Final Lower Band
        if basic_lb_4h[i] > supertrend_4h[i-1] or close_4h[i-1] < supertrend_4h[i-1]:
            final_lb = basic_lb_4h[i]
        else:
            final_lb = supertrend_4h[i-1]
            
        # Supertrend
        if i == 1:
            supertrend_4h[i] = final_ub
            direction_4h[i] = -1
        else:
            if supertrend_4h[i-1] == supertrend_4h[i-1]:  # Not NaN
                if supertrend_4h[i-1] == final_ub:
                    if close_4h[i] <= final_ub:
                        supertrend_4h[i] = final_ub
                    else:
                        supertrend_4h[i] = final_lb
                        direction_4h[i] = 1
                else:
                    if close_4h[i] >= final_lb:
                        supertrend_4h[i] = final_lb
                        direction_4h[i] = 1
                    else:
                        supertrend_4h[i] = final_ub
                        direction_4h[i] = -1
            else:
                supertrend_4h[i] = final_ub
                direction_4h[i] = -1
    
    # Calculate 1d Supertrend (ATR=10, mult=3.0)
    tr1_1d = high_1d - low_1d
    tr2_1d = np.abs(high_1d - np.roll(close_1d, 1))
    tr3_1d = np.abs(low_1d - np.roll(close_1d, 1))
    tr_1d = np.maximum(tr1_1d, np.maximum(tr2_1d, tr3_1d))
    tr_1d[0] = tr1_1d[0]
    atr_1d = pd.Series(tr_1d).rolling(window=atr_period, min_periods=atr_period).mean().values
    
    basic_ub_1d = (high_1d + low_1d) / 2 + multiplier * atr_1d
    basic_lb_1d = (high_1d + low_1d) / 2 - multiplier * atr_1d
    
    supertrend_1d = np.full_like(close_1d, np.nan, dtype=float)
    direction_1d = np.full_like(close_1d, 1, dtype=float)
    
    for i in range(1, len(close_1d)):
        if i < atr_period:
            continue
            
        if basic_ub_1d[i] < supertrend_1d[i-1] or close_1d[i-1] > supertrend_1d[i-1]:
            final_ub = basic_ub_1d[i]
        else:
            final_ub = supertrend_1d[i-1]
            
        if basic_lb_1d[i] > supertrend_1d[i-1] or close_1d[i-1] < supertrend_1d[i-1]:
            final_lb = basic_lb_1d[i]
        else:
            final_lb = supertrend_1d[i-1]
            
        if i == 1:
            supertrend_1d[i] = final_ub
            direction_1d[i] = -1
        else:
            if supertrend_1d[i-1] == supertrend_1d[i-1]:
                if supertrend_1d[i-1] == final_ub:
                    if close_1d[i] <= final_ub:
                        supertrend_1d[i] = final_ub
                    else:
                        supertrend_1d[i] = final_lb
                        direction_1d[i] = 1
                else:
                    if close_1d[i] >= final_lb:
                        supertrend_1d[i] = final_lb
                        direction_1d[i] = 1
                    else:
                        supertrend_1d[i] = final_ub
                        direction_1d[i] = -1
            else:
                supertrend_1d[i] = final_ub
                direction_1d[i] = -1
    
    # Calculate 1h RSI(14)
    delta = pd.Series(close).diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14, min_periods=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14, min_periods=14).mean()
    rs = gain / loss
    rsi = 100 - (100 / (1 + rs))
    rsi = rsi.fillna(50).values
    
    # Align HTF Supertrend and direction to 1h timeframe
    supertrend_4h_aligned = align_htf_to_ltf(prices, df_4h, supertrend_4h)
    direction_4h_aligned = align_htf_to_ltf(prices, df_4h, direction_4h)
    supertrend_1d_aligned = align_htf_to_ltf(prices, df_1d, supertrend_1d)
    direction_1d_aligned = align_htf_to_ltf(prices, df_1d, direction_1d)
    
    # Pre-compute volume confirmation (20-period average for 1h)
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    # Pre-compute session filter (08-20 UTC)
    hours = pd.DatetimeIndex(open_time).hour
    in_session = (hours >= 8) & (hours <= 20)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(100, n):
        # Skip if any required data is invalid or outside session
        if (np.isnan(supertrend_4h_aligned[i]) or np.isnan(direction_4h_aligned[i]) or
            np.isnan(supertrend_1d_aligned[i]) or np.isnan(direction_1d_aligned[i]) or
            np.isnan(rsi[i]) or np.isnan(vol_ma_20[i]) or not in_session[i] or
            vol_ma_20[i] <= 0):
            signals[i] = 0.0
            continue
        
        # Volume confirmation: current 1h volume > 1.3x average 1h volume
        volume_confirmed = volume[i] > 1.3 * vol_ma_20[i]
        
        # Fixed position size to minimize fee churn
        position_size = 0.20
        
        if position == 1:  # Long position
            # Exit when either 4h or 1d trend turns down
            if (direction_4h_aligned[i] == -1 or direction_1d_aligned[i] == -1):
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = position_size
                
        elif position == -1:  # Short position
            # Exit when either 4h or 1d trend turns up
            if (direction_4h_aligned[i] == 1 or direction_1d_aligned[i] == 1):
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -position_size
        else:  # Flat
            # Trend-following entry with volume confirmation
            if volume_confirmed:
                # Long when both 4h and 1d are uptrend and RSI > 50
                if (direction_4h_aligned[i] == 1 and direction_1d_aligned[i] == 1 and rsi[i] > 50):
                    position = 1
                    signals[i] = position_size
                # Short when both 4h and 1d are downtrend and RSI < 50
                elif (direction_4h_aligned[i] == -1 and direction_1d_aligned[i] == -1 and rsi[i] < 50):
                    position = -1
                    signals[i] = -position_size
    
    return signals
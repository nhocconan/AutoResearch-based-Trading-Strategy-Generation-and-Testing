#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h Williams %R + 1d ADX regime filter
# Williams %R(14): overbought > -20, oversold < -80
# ADX > 25 indicates trending market (use 1d ADX for regime)
# In trending regime (ADX > 25): trade pullbacks - long when %R crosses above -80 from below, short when crosses below -20 from above
# In ranging regime (ADX <= 25): mean reversion - long when %R < -80 and turning up, short when %R > -20 and turning down
# Uses 1d ADX(14) for regime detection, 12h for Williams %R and entries
# Position size 0.25 to limit drawdown
# Target: 50-150 total trades over 4 years (12-37/year) to minimize fee drag

name = "12h_1d_williams_r_adx_regime_v1"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    
    # Load 1d data ONCE before loop for ADX
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    # Calculate 1d ADX(14) for regime detection
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # True Range
    tr = np.zeros(len(df_1d))
    tr[0] = high_1d[0] - low_1d[0]
    for i in range(1, len(df_1d)):
        tr0 = high_1d[i] - low_1d[i]
        tr1 = abs(high_1d[i] - close_1d[i-1])
        tr2 = abs(low_1d[i] - close_1d[i-1])
        tr[i] = max(tr0, tr1, tr2)
    
    # Directional Movement
    plus_dm = np.zeros(len(df_1d))
    minus_dm = np.zeros(len(df_1d))
    for i in range(1, len(df_1d)):
        up_move = high_1d[i] - high_1d[i-1]
        down_move = low_1d[i-1] - low_1d[i]
        if up_move > down_move and up_move > 0:
            plus_dm[i] = up_move
        else:
            plus_dm[i] = 0
        if down_move > up_move and down_move > 0:
            minus_dm[i] = down_move
        else:
            minus_dm[i] = 0
    
    # Smoothed DM and TR (Wilder's smoothing)
    def wilders_smoothing(data, period):
        result = np.full(len(data), np.nan)
        if len(data) < period:
            return result
        result[period-1] = np.nansum(data[:period])
        for i in range(period, len(data)):
            result[i] = result[i-1] - (result[i-1] / period) + data[i]
        return result
    
    # Calculate smoothed values
    tr_14 = wilders_smoothing(tr, 14)
    plus_dm_14 = wilders_smoothing(plus_dm, 14)
    minus_dm_14 = wilders_smoothing(minus_dm, 14)
    
    # Calculate DI and DX
    plus_di_14 = np.full(len(df_1d), np.nan)
    minus_di_14 = np.full(len(df_1d), np.nan)
    dx_14 = np.full(len(df_1d), np.nan)
    
    for i in range(14, len(df_1d)):
        if tr_14[i] != 0:
            plus_di_14[i] = (plus_dm_14[i] / tr_14[i]) * 100
            minus_di_14[i] = (minus_dm_14[i] / tr_14[i]) * 100
            if (plus_di_14[i] + minus_di_14[i]) != 0:
                dx_14[i] = (abs(plus_di_14[i] - minus_di_14[i]) / (plus_di_14[i] + minus_di_14[i])) * 100
    
    # Calculate ADX (smoothed DX)
    adx_14 = wilders_smoothing(dx_14, 14)
    
    # Align 1d ADX to 12h timeframe
    adx_14_12h = align_htf_to_ltf(prices, df_1d, adx_14)
    
    # Calculate Williams %R on 12h
    def calculate_williams_r(high, low, close, period):
        highest_high = np.full(len(high), np.nan)
        lowest_low = np.full(len(low), np.nan)
        for i in range(period-1, len(high)):
            highest_high[i] = np.max(high[i-(period-1):i+1])
            lowest_low[i] = np.min(low[i-(period-1):i+1])
        wr = np.full(len(high), np.nan)
        for i in range(period-1, len(high)):
            if highest_high[i] - lowest_low[i] != 0:
                wr[i] = (highest_high[i] - close[i]) / (highest_high[i] - lowest_low[i]) * -100
        return wr
    
    wr = calculate_williams_r(high, low, close, 14)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(30, n):  # Start after warmup
        # Skip if any required data is invalid
        if (np.isnan(adx_14_12h[i]) or 
            np.isnan(wr[i])):
            signals[i] = 0.0
            continue
        
        adx = adx_14_12h[i]
        wr_val = wr[i]
        
        if position == 1:  # Long position
            # Exit conditions
            if adx > 25:  # Trending regime
                # Exit when Williams %R crosses below -50 (momentum loss)
                if wr_val < -50:
                    position = 0
                    signals[i] = 0.0
                else:
                    signals[i] = 0.25
            else:  # Ranging regime
                # Exit when Williams %R returns to mean (-50)
                if wr_val > -50:
                    position = 0
                    signals[i] = 0.0
                else:
                    signals[i] = 0.25
                    
        elif position == -1:  # Short position
            # Exit conditions
            if adx > 25:  # Trending regime
                # Exit when Williams %R crosses above -50 (momentum loss)
                if wr_val > -50:
                    position = 0
                    signals[i] = 0.0
                else:
                    signals[i] = -0.25
            else:  # Ranging regime
                # Exit when Williams %R returns to mean (-50)
                if wr_val < -50:
                    position = 0
                    signals[i] = 0.0
                else:
                    signals[i] = -0.25
        else:  # Flat
            # Entry logic based on regime
            if adx > 25:  # Trending regime - trade pullbacks
                # Go long when Williams %R crosses above -80 from below (oversold bounce)
                # Go short when Williams %R crosses below -20 from above (overbought rejection)
                if i > 30:
                    wr_prev = wr[i-1]
                    if wr_val > -80 and wr_prev <= -80:
                        position = 1
                        signals[i] = 0.25
                    elif wr_val < -20 and wr_prev >= -20:
                        position = -1
                        signals[i] = -0.25
            else:  # Ranging regime - mean reversion
                # Go long when Williams %R is deeply oversold (< -80)
                # Go short when Williams %R is deeply overbought (> -20)
                if wr_val < -80:
                    position = 1
                    signals[i] = 0.25
                elif wr_val > -20:
                    position = -1
                    signals[i] = -0.25
    
    return signals
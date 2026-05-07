#!/usr/bin/env python3

name = "12h_Camarilla_R3S3_Breakout_1dTrend_Volume"
timeframe = "12h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def calculate_pivot_levels(high, low, close):
    """Calculate Camarilla pivot levels: R3, R2, R1, PP, S1, S2, S3"""
    H, L, C = high, low, close
    range_val = H - L
    if range_val == 0:
        return C, C, C, C, C, C, C
    PP = (H + L + C) / 3
    R1 = C + (range_val * 1.1 / 12)
    R2 = C + (range_val * 1.1 / 6)
    R3 = C + (range_val * 1.1 / 4)
    S1 = C - (range_val * 1.1 / 12)
    S2 = C - (range_val * 1.1 / 6)
    S3 = C - (range_val * 1.1 / 4)
    return R3, R2, R1, PP, S1, S2, S3

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1d data for trend filter and pivot levels
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    # Calculate 1d EMA34 for trend filter
    close_1d = df_1d['close'].values
    ema_34_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    
    # Calculate 1d pivot levels
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    R3, R2, R1, PP, S1, S2, S3 = calculate_pivot_levels(high_1d, low_1d, close_1d)
    
    # Align 1d data to 12h timeframe
    ema_34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    R3_aligned = align_htf_to_ltf(prices, df_1d, R3)
    R2_aligned = align_htf_to_ltf(prices, df_1d, R2)
    R1_aligned = align_htf_to_ltf(prices, df_1d, R1)
    PP_aligned = align_htf_to_ltf(prices, df_1d, PP)
    S1_aligned = align_htf_to_ltf(prices, df_1d, S1)
    S2_aligned = align_htf_to_ltf(prices, df_1d, S2)
    S3_aligned = align_htf_to_ltf(prices, df_1d, S3)
    close_1d_aligned = align_htf_to_ltf(prices, df_1d, close_1d)
    
    # Volume filter: current volume > 1.5x 20-period average (12h)
    vol_ma_20 = np.full(n, np.nan)
    for i in range(20, n):
        vol_ma_20[i] = np.mean(volume[i-20:i])
    vol_filter = volume > (1.5 * vol_ma_20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    bars_since_last_trade = 0
    cooldown_bars = 4  # ~2 days for 12h to reduce trades (48h/12h = 4)
    
    start_idx = max(50, 20)
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(ema_34_1d_aligned[i]) or 
            np.isnan(R3_aligned[i]) or 
            np.isnan(S3_aligned[i]) or 
            np.isnan(vol_ma_20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
                bars_since_last_trade = 0
            else:
                bars_since_last_trade += 1
            continue
        
        bars_since_last_trade += 1
        
        # Determine 1d trend direction
        trend_1d_up = close_1d_aligned[i] > ema_34_1d_aligned[i]
        trend_1d_down = close_1d_aligned[i] < ema_34_1d_aligned[i]
        
        if position == 0 and bars_since_last_trade >= cooldown_bars:
            # Long: Price breaks above R3 with volume in 1d uptrend
            if (close[i] > R3_aligned[i] and 
                trend_1d_up and 
                vol_filter[i]):
                signals[i] = 0.25
                position = 1
                bars_since_last_trade = 0
            # Short: Price breaks below S3 with volume in 1d downtrend
            elif (close[i] < S3_aligned[i] and 
                  trend_1d_down and 
                  vol_filter[i]):
                signals[i] = -0.25
                position = -1
                bars_since_last_trade = 0
        elif position == 1:
            # Exit: Price re-enters below R1 or trend change
            if (close[i] < R1_aligned[i]) or not trend_1d_up:
                signals[i] = 0.0
                position = 0
                bars_since_last_trade = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit: Price re-enters above S1 or trend change
            if (close[i] > S1_aligned[i]) or not trend_1d_down:
                signals[i] = 0.0
                position = 0
                bars_since_last_trade = 0
            else:
                signals[i] = -0.25
    
    return signals

# Hypothesis: Camarilla R3/S3 levels act as strong support/resistance. Breaking these levels with volume confirmation in the direction of the 1d trend captures significant moves. The 1d EMA34 filter ensures we only trade with the higher timeframe trend. Volume filter avoids false breakouts. Cooldown reduces trade frequency. Target: 20-40 trades/year. Works in bull markets by buying R3 breakouts in uptrends and in bear markets by selling S3 breakdowns in downtrends. 12h timeframe balances signal quality and trade frequency. Camarilla levels provide precise entry/exit points while EMA34 establishes the trend direction. Volume ensures participation. This avoids overtrading by requiring multiple confirmations.
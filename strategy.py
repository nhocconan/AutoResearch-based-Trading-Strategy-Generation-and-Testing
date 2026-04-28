#!/usr/bin/env python3
"""
4h_MFI_Overbought_Oversold_1wTrend
Hypothesis: Money Flow Index (MFI) identifies overbought (>80) and oversold (<20) conditions.
In strong weekly trends, these extremes often reverse. Long when MFI crosses below 20 with
weekly uptrend; short when MFI crosses above 80 with weekly downtrend. Uses volume-weighted
RSI concept to capture exhaustion points with lower frequency than RSI alone.
Designed for 20-30 trades/year to minimize fee drag while capturing mean reversion in trends.
Works in both bull and bear by aligning with weekly trend direction.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 30:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get weekly data for trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 30:
        return np.zeros(n)
    
    # Calculate weekly EMA50 for trend filter
    close_1w = df_1w['close'].values
    ema_50_1w = pd.Series(close_1w).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_50_1w)
    
    # Calculate MFI (Money Flow Index) - 14 period
    typical_price = (high + low + close) / 3
    raw_money_flow = typical_price * volume
    
    # Positive and negative money flow
    positive_flow = np.where(typical_price > np.roll(typical_price, 1), raw_money_flow, 0)
    negative_flow = np.where(typical_price < np.roll(typical_price, 1), raw_money_flow, 0)
    
    # Handle first element
    positive_flow[0] = 0
    negative_flow[0] = 0
    
    # Calculate money flow ratio
    positive_mf = pd.Series(positive_flow).rolling(window=14, min_periods=14).sum().values
    negative_mf = pd.Series(negative_flow).rolling(window=14, min_periods=14).sum().values
    
    # Avoid division by zero
    money_flow_ratio = np.divide(positive_mf, negative_mf, 
                                out=np.full_like(positive_mf, 1.0), 
                                where=negative_mf!=0)
    mfi = 100 - (100 / (1 + money_flow_ratio))
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 14  # Wait for MFI to stabilize
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(mfi[i]) or 
            np.isnan(mfi[i-1]) or
            np.isnan(ema_50_1w_aligned[i])):
            signals[i] = 0.0
            continue
        
        # MFI cross signals
        mfi_cross_up_20 = (mfi[i-1] <= 20) and (mfi[i] > 20)
        mfi_cross_down_80 = (mfi[i-1] >= 80) and (mfi[i] < 80)
        
        # Trend filter from weekly EMA50
        uptrend = close[i] > ema_50_1w_aligned[i]
        downtrend = close[i] < ema_50_1w_aligned[i]
        
        # Entry conditions
        long_entry = mfi_cross_up_20 and uptrend
        short_entry = mfi_cross_down_80 and downtrend
        
        # Exit on opposite extreme
        long_exit = mfi_cross_down_80
        short_exit = mfi_cross_up_20
        
        if long_entry and position <= 0:
            signals[i] = 0.25
            position = 1
        elif short_entry and position >= 0:
            signals[i] = -0.25
            position = -1
        elif long_exit and position == 1:
            signals[i] = -0.25  # Reverse to short
            position = -1
        elif short_exit and position == -1:
            signals[i] = 0.25   # Reverse to long
            position = 1
        else:
            # Hold current position
            if position == 1:
                signals[i] = 0.25
            elif position == -1:
                signals[i] = -0.25
            else:
                signals[i] = 0.0
    
    return signals

name = "4h_MFI_Overbought_Oversold_1wTrend"
timeframe = "4h"
leverage = 1.0
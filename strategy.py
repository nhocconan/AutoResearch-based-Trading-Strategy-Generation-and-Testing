#!/usr/bin/env python3
"""
1d_WeeklyTrend_Filter_Refined
1d strategy using weekly trend filter with daily price action confirmation.
- Long: Weekly EMA21 trending up + daily price crosses above daily EMA21 with volume confirmation
- Short: Weekly EMA21 trending down + daily price crosses below daily EMA21 with volume confirmation
- Exit: Opposite signal
Designed for ~10-20 trades/year per symbol (40-80 total over 4 years)
Uses weekly trend to avoid whipsaws in ranging markets
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Get weekly data for trend filter
    df_weekly = get_htf_data(prices, '1w')
    
    # Calculate weekly EMA21 for trend direction
    weekly_close = df_weekly['close'].values
    ema_21_weekly = pd.Series(weekly_close).ewm(span=21, adjust=False, min_periods=21).mean().values
    ema_21_weekly_aligned = align_htf_to_ltf(prices, df_weekly, ema_21_weekly)
    
    # Calculate daily EMA21 for entry signals
    ema_21_daily = pd.Series(close).ewm(span=21, adjust=False, min_periods=21).mean().values
    
    # Volume confirmation (1.5x 20-day average)
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 30  # need 21 for EMA + buffer
    
    for i in range(start_idx, n):
        # Skip if any required data is not available
        if (np.isnan(ema_21_weekly_aligned[i]) or 
            np.isnan(ema_21_daily[i]) or 
            np.isnan(vol_ma[i])):
            signals[i] = 0.0
            continue
        
        # Weekly trend direction
        weekly_uptrend = ema_21_weekly_aligned[i] > ema_21_weekly_aligned[i-1]
        weekly_downtrend = ema_21_weekly_aligned[i] < ema_21_weekly_aligned[i-1]
        
        # Daily price relative to EMA21
        price_above_ema = close[i] > ema_21_daily[i]
        price_below_ema = close[i] < ema_21_daily[i]
        
        # Volume confirmation
        volume_confirmed = volume[i] > 1.5 * vol_ma[i]
        
        if position == 0:
            # Long: weekly uptrend + price crosses above daily EMA21 + volume
            if weekly_uptrend and price_above_ema and volume_confirmed and close[i-1] <= ema_21_daily[i-1]:
                signals[i] = 0.25
                position = 1
            # Short: weekly downtrend + price crosses below daily EMA21 + volume
            elif weekly_downtrend and price_below_ema and volume_confirmed and close[i-1] >= ema_21_daily[i-1]:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Long exit: weekly trend turns down OR price crosses below daily EMA21
            if not weekly_uptrend or (price_below_ema and close[i-1] >= ema_21_daily[i-1]):
                signals[i] = -0.25  # reverse to short
                position = -1
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: weekly trend turns up OR price crosses above daily EMA21
            if not weekly_downtrend or (price_above_ema and close[i-1] <= ema_21_daily[i-1]):
                signals[i] = 0.25  # reverse to long
                position = 1
            else:
                signals[i] = -0.25
    
    return signals

name = "1d_WeeklyTrend_Filter_Refined"
timeframe = "1d"
leverage = 1.0
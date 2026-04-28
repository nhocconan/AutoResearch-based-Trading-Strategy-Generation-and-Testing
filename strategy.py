#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get weekly data once for HTF context
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 20:
        return np.zeros(n)
    
    # Weekly high/low/close for calculations
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    
    # Calculate weekly range for pivot calculations
    weekly_range = high_1w - low_1w
    
    # Weekly Camarilla pivot levels (based on previous week)
    camarilla_r4 = close_1w + weekly_range * 1.1 / 2
    camarilla_s4 = close_1w - weekly_range * 1.1 / 2
    
    # Weekly EMA21 for trend
    ema_21_1w = pd.Series(close_1w).ewm(span=21, adjust=False, min_periods=21).mean().values
    
    # Align Weekly Camarilla levels and weekly EMA to daily timeframe
    r4_aligned = align_htf_to_ltf(prices, df_1w, camarilla_r4)
    s4_aligned = align_htf_to_ltf(prices, df_1w, camarilla_s4)
    ema_21_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_21_1w)
    
    # Hour filter: 8-20 UTC (most active trading hours)
    hours = pd.DatetimeIndex(prices['open_time']).hour
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 50  # Wait for sufficient warmup
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(r4_aligned[i]) or np.isnan(s4_aligned[i]) or
            np.isnan(ema_21_1w_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Session filter: only trade 8-20 UTC
        hour = hours[i]
        in_session = 8 <= hour <= 20
        
        if not in_session:
            # Outside session: flatten position
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue
        
        # Weekly trend filter: price above/below weekly EMA21
        price_above_weekly_ema = close[i] > ema_21_1w_aligned[i]
        price_below_weekly_ema = close[i] < ema_21_1w_aligned[i]
        
        # Price relative to weekly Camarilla levels
        price_above_weekly_r4 = close[i] > r4_aligned[i]
        price_below_weekly_s4 = close[i] < s4_aligned[i]
        
        # Entry conditions: 
        # Long: price breaks above weekly R4 with weekly uptrend
        # Short: price breaks below weekly S4 with weekly downtrend
        long_entry = price_above_weekly_r4 and price_above_weekly_ema
        short_entry = price_below_weekly_s4 and price_below_weekly_ema
        
        # Exit conditions: price returns to opposite weekly S4/R4 levels
        long_exit = price_below_weekly_s4
        short_exit = price_above_weekly_r4
        
        if long_entry and position <= 0:
            signals[i] = 0.25
            position = 1
        elif short_entry and position >= 0:
            signals[i] = -0.25
            position = -1
        elif long_exit and position == 1:
            signals[i] = 0.0
            position = 0
        elif short_exit and position == -1:
            signals[i] = 0.0
            position = 0
        else:
            # Hold current position
            if position == 1:
                signals[i] = 0.25
            elif position == -1:
                signals[i] = -0.25
            else:
                signals[i] = 0.0
    
    return signals

name = "1d_WeeklyCamarilla_R4S4_Trend_EMA21"
timeframe = "1d"
leverage = 1.0
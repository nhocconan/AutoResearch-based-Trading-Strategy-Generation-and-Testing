# 12h_Camarilla_R4S4_Breakout_WeeklyEMA21
# Hypothesis: Use 12h timeframe with Camarilla R4/S4 levels from daily pivots and weekly EMA21 trend filter.
# Targets 12-37 trades/year (50-150 total) by combining daily pivot breakouts with weekly trend alignment.
# Works in bull/bear: Weekly EMA21 filters trend direction, reducing false breakouts in ranging markets.
# Volume confirmation ensures breakout validity. Designed for low frequency to minimize fee drag.

#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 60:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get daily data once for Camarilla pivots
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    # Get weekly data once for trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 20:
        return np.zeros(n)
    
    # Daily OHLC for pivot calculation (previous day's values)
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate daily range
    daily_range = high_1d - low_1d
    
    # Camarilla R4 and S4 levels (based on previous day)
    camarilla_r4 = close_1d + daily_range * 1.1 / 2
    camarilla_s4 = close_1d - daily_range * 1.1 / 2
    
    # Weekly EMA21 for trend filter
    close_1w = df_1w['close'].values
    ema_21_1w = pd.Series(close_1w).ewm(span=21, adjust=False, min_periods=21).mean().values
    
    # Align HTF indicators to 12h timeframe (only use completed bars)
    r4_aligned = align_htf_to_ltf(prices, df_1d, camarilla_r4)
    s4_aligned = align_htf_to_ltf(prices, df_1d, camarilla_s4)
    ema_21_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_21_1w)
    
    # Volume filter: 20-period moving average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    # Session filter: 8-20 UTC (most active trading hours)
    hours = pd.DatetimeIndex(prices['open_time']).hour
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 60  # Wait for sufficient warmup
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(r4_aligned[i]) or np.isnan(s4_aligned[i]) or 
            np.isnan(ema_21_1w_aligned[i]) or np.isnan(vol_ma[i])):
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
        
        # Volume filter: above average volume
        vol_filter = volume[i] > vol_ma[i]
        
        # Weekly trend filter: price above/below weekly EMA21
        price_above_weekly_ema = close[i] > ema_21_1w_aligned[i]
        price_below_weekly_ema = close[i] < ema_21_1w_aligned[i]
        
        # Entry conditions: 
        # Long: price breaks above R4 with volume and weekly uptrend
        # Short: price breaks below S4 with volume and weekly downtrend
        long_entry = (close[i] > r4_aligned[i]) and price_above_weekly_ema and vol_filter
        short_entry = (close[i] < s4_aligned[i]) and price_below_weekly_ema and vol_filter
        
        # Exit conditions: price returns to opposite S4/R4 levels or weekly trend reversal
        long_exit = (close[i] < s4_aligned[i]) or (not price_above_weekly_ema)
        short_exit = (close[i] > r4_aligned[i]) or (not price_below_weekly_ema)
        
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

name = "12h_Camarilla_R4S4_Breakout_WeeklyEMA21"
timeframe = "12h"
leverage = 1.0
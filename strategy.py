#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get daily data once for HTF context
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # Daily high/low/close for calculations
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate daily range for pivot calculations
    daily_range = high_1d - low_1d
    
    # Calculate weekly data for context
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 20:
        return np.zeros(n)
    
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    
    # Calculate weekly range for pivot calculations
    weekly_range = high_1w - low_1w
    
    # Calculate 12-period ATR for volatility filter
    tr1 = high[1:] - low[1:]
    tr2 = np.abs(high[1:] - close[:-1])
    tr3 = np.abs(low[1:] - close[:-1])
    tr = np.concatenate([[np.inf], np.maximum(tr1, np.maximum(tr2, tr3))])
    atr = pd.Series(tr).rolling(window=12, min_periods=12).mean().values
    
    # Camarilla pivot levels (based on previous day)
    camarilla_r4 = close_1d + daily_range * 1.1 / 2
    camarilla_s4 = close_1d - daily_range * 1.1 / 2
    
    # Weekly trend: EMA21
    ema_21_1w = pd.Series(close_1w).ewm(span=21, adjust=False, min_periods=21).mean().values
    
    # Align Camarilla levels and weekly EMA to 12h timeframe
    r4_aligned = align_htf_to_ltf(prices, df_1d, camarilla_r4)
    s4_aligned = align_htf_to_ltf(prices, df_1d, camarilla_s4)
    ema_21_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_21_1w)
    
    # Volume filter: above average volume (24-period)
    vol_ma = pd.Series(volume).rolling(window=24, min_periods=24).mean().values
    
    # Hour filter: 0-23 UTC (all hours for 12h timeframe)
    hours = pd.DatetimeIndex(prices['open_time']).hour
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 100  # Wait for sufficient warmup
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(r4_aligned[i]) or np.isnan(s4_aligned[i]) or 
            np.isnan(ema_21_1w_aligned[i]) or np.isnan(vol_ma[i]) or np.isnan(atr[i])):
            signals[i] = 0.0
            continue
        
        # Session filter: all hours for 12h timeframe (no restriction)
        hour = hours[i]
        in_session = True  # 12h timeframe trades all hours
        
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
        
        # Volatility filter: ATR > 0 (always true) and not extremely low
        vol_filter_low = atr[i] > 0
        
        # Weekly trend filter: price above/below weekly EMA21
        price_above_weekly_ema = close[i] > ema_21_1w_aligned[i]
        price_below_weekly_ema = close[i] < ema_21_1w_aligned[i]
        
        # Entry conditions: 
        # Long: price breaks above R4 with volume and weekly uptrend
        # Short: price breaks below S4 with volume and weekly downtrend
        long_entry = (close[i] > r4_aligned[i]) and price_above_weekly_ema and vol_filter and vol_filter_low
        short_entry = (close[i] < s4_aligned[i]) and price_below_weekly_ema and vol_filter and vol_filter_low
        
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
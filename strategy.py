#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "12h_Camarilla_Pivot_1dVolume_Trend"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    
    # Get 1d data for Camarilla pivots, volume, and trend
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    # Calculate 1d Camarilla pivot levels (based on previous day's range)
    # Camarilla: H4 = Close + 1.5*(High-Low)*1.1/2, L4 = Close - 1.5*(High-Low)*1.1/2
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    camarilla_h4 = np.full(len(df_1d), np.nan)
    camarilla_l4 = np.full(len(df_1d), np.nan)
    
    for i in range(1, len(df_1d)):  # need previous day's data
        high_prev = high_1d[i-1]
        low_prev = low_1d[i-1]
        close_prev = close_1d[i-1]
        range_prev = high_prev - low_prev
        
        camarilla_h4[i] = close_prev + 1.5 * range_prev * 1.1 / 2
        camarilla_l4[i] = close_prev - 1.5 * range_prev * 1.1 / 2
    
    # Calculate 1d volume average (20-period)
    vol_1d = df_1d['volume'].values
    vol_avg_20 = np.full(len(df_1d), np.nan)
    for i in range(len(df_1d)):
        if i >= 19:
            vol_avg_20[i] = np.mean(vol_1d[i-19:i+1])
    
    # Calculate 1d EMA50 for trend filter
    close_series = pd.Series(close_1d)
    ema_50 = close_series.ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Align 1d data to 12h timeframe
    camarilla_h4_aligned = align_htf_to_ltf(prices, df_1d, camarilla_h4)
    camarilla_l4_aligned = align_htf_to_ltf(prices, df_1d, camarilla_l4)
    vol_avg_20_aligned = align_htf_to_ltf(prices, df_1d, vol_avg_20)
    ema_50_aligned = align_htf_to_ltf(prices, df_1d, ema_50)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(20, 50)  # warmup for indicators
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(camarilla_h4_aligned[i]) or np.isnan(camarilla_l4_aligned[i]) or 
            np.isnan(vol_avg_20_aligned[i]) or np.isnan(ema_50_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Get current 1d bar's data (last completed 1d bar)
        idx_1d = 0
        while idx_1d < len(df_1d) and df_1d.iloc[idx_1d]['open_time'] <= prices.iloc[i]['open_time']:
            idx_1d += 1
        idx_1d -= 1  # last completed 1d bar
        
        if idx_1d < 0:
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        camarilla_h4_current = camarilla_h4[idx_1d]
        camarilla_l4_current = camarilla_l4[idx_1d]
        vol_avg_20_current = vol_avg_20[idx_1d]
        ema_50_current = ema_50[idx_1d]
        
        if (np.isnan(camarilla_h4_current) or np.isnan(camarilla_l4_current) or 
            np.isnan(vol_avg_20_current) or np.isnan(ema_50_current)):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Volume confirmation: current 1d volume > 1.5x 20-period average
        vol_current = df_1d['volume'].iloc[idx_1d]
        vol_confirmed = vol_current > 1.5 * vol_avg_20_current
        
        # Trend filter: price above/below EMA50
        price_above_ema = close_1d[idx_1d] > ema_50_current
        price_below_ema = close_1d[idx_1d] < ema_50_current
        
        # Trading logic
        if position == 0:
            # Look for entry
            if vol_confirmed:
                # Long when price touches L4 and in uptrend
                if close[i] <= camarilla_l4_aligned[i] and price_above_ema:
                    signals[i] = 0.25
                    position = 1
                # Short when price touches H4 and in downtrend
                elif close[i] >= camarilla_h4_aligned[i] and price_below_ema:
                    signals[i] = -0.25
                    position = -1
        elif position == 1:
            # Manage long position
            exit_signal = False
            # Exit when price reaches H4 or trend changes
            if close[i] >= camarilla_h4_aligned[i]:
                exit_signal = True
            elif not price_above_ema:  # trend turned against position
                exit_signal = True
            elif not vol_confirmed:  # volume confirmation lost
                exit_signal = True
            
            if exit_signal:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Manage short position
            exit_signal = False
            # Exit when price reaches L4 or trend changes
            if close[i] <= camarilla_l4_aligned[i]:
                exit_signal = True
            elif not price_below_ema:  # trend turned against position
                exit_signal = True
            elif not vol_confirmed:  # volume confirmation lost
                exit_signal = True
            
            if exit_signal:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals
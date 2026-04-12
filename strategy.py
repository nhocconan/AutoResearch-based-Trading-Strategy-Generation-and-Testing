#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Hypothesis: 12h Camarilla pivot breakout with 1d EMA200 trend filter + volume confirmation
    # Uses 1d EMA200 for trend filter: only take breakouts in direction of 1d trend
    # Volume confirmation: volume > 2.0 * 20-period average to filter false breakouts
    # Discrete sizing 0.25 to minimize fee churn. Target: 12-37 trades/year per symbol.
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1d data for trend filter and Camarilla pivots
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 200:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    
    # Calculate 1d EMA200 for trend filter
    ema200_1d = pd.Series(close_1d).ewm(span=200, adjust=False, min_periods=200).mean().values
    ema200_1d_aligned = align_htf_to_ltf(prices, df_1d, ema200_1d)
    
    # Calculate 1d Camarilla pivot levels (based on previous day's range)
    camarilla_h4 = np.full(n, np.nan)  # Resistance 4 (strongest resistance)
    camarilla_l4 = np.full(n, np.nan)  # Support 4 (strongest support)
    camarilla_h3 = np.full(n, np.nan)  # Resistance 3
    camarilla_l3 = np.full(n, np.nan)  # Support 3
    
    # Calculate pivots from previous 1d bar (to avoid look-ahead)
    for i in range(1, len(df_1d)):
        # Previous day's high, low, close
        ph = high_1d[i-1]
        pl = low_1d[i-1]
        pc = close_1d[i-1]
        
        # Camarilla calculations
        range_val = ph - pl
        camarilla_h4[i] = pc + (range_val * 1.1 / 2)
        camarilla_l4[i] = pc - (range_val * 1.1 / 2)
        camarilla_h3[i] = pc + (range_val * 1.1 / 4)
        camarilla_l3[i] = pc - (range_val * 1.1 / 4)
    
    # Align Camarilla levels to 12h timeframe
    camarilla_h4_aligned = align_htf_to_ltf(prices, df_1d, camarilla_h4)
    camarilla_l4_aligned = align_htf_to_ltf(prices, df_1d, camarilla_l4)
    camarilla_h3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_h3)
    camarilla_l3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_l3)
    
    # Volume confirmation: volume > 2.0 * 20-period average
    vol_ma = np.full(n, np.nan)
    for i in range(20, n):
        vol_ma[i] = np.mean(volume[i-20:i])
    volume_spike = volume > (2.0 * vol_ma)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(50, n):
        # Skip if data not ready
        if (np.isnan(ema200_1d_aligned[i]) or np.isnan(camarilla_h4_aligned[i]) or 
            np.isnan(camarilla_l4_aligned[i]) or np.isnan(vol_ma[i])):
            signals[i] = 0.0
            continue
        
        # Determine 1d trend
        bullish_trend = close[i] > ema200_1d_aligned[i]
        bearish_trend = close[i] < ema200_1d_aligned[i]
        
        # Entry logic: Camarilla breakout with volume and trend filter
        long_entry = False
        short_entry = False
        
        # Long breakout: price breaks above H4 resistance in bullish trend
        if bullish_trend:
            long_entry = (close[i] > camarilla_h4_aligned[i]) and volume_spike[i]
        # Short breakout: price breaks below L4 support in bearish trend
        elif bearish_trend:
            short_entry = (close[i] < camarilla_l4_aligned[i]) and volume_spike[i]
        
        # Exit logic: opposite Camarilla level or trend reversal
        long_exit = (bearish_trend and close[i] < camarilla_l3_aligned[i]) or (not bullish_trend and not bearish_trend)
        short_exit = (bullish_trend and close[i] > camarilla_h3_aligned[i]) or (not bullish_trend and not bearish_trend)
        
        if long_entry and position != 1:
            position = 1
            signals[i] = 0.25
        elif short_entry and position != -1:
            position = -1
            signals[i] = -0.25
        elif position == 1 and long_exit:
            position = 0
            signals[i] = 0.0
        elif position == -1 and short_exit:
            position = 0
            signals[i] = 0.0
        else:
            # Hold current position
            if position == 1:
                signals[i] = 0.25
            elif position == -1:
                signals[i] = -0.25
            else:
                signals[i] = 0.0
    
    return signals

name = "12h_1d_camarilla_breakout_trend_volume_v1"
timeframe = "12h"
leverage = 1.0
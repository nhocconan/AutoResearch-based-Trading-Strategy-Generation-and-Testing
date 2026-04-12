#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Hypothesis: 6h Camarilla pivot breakout with 1w EMA200 trend filter + volume confirmation
    # Uses 1w EMA200 for trend filter: only take breakouts in direction of weekly trend
    # Breakout logic: price breaks above R4 (bullish) or below S4 (bearish) Camarilla levels from prior 1d
    # Volume confirmation: volume > 2.0 * 20-period average to filter false breakouts
    # Discrete sizing 0.25 to minimize fee churn. Target: 15-30 trades/year per symbol.
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1w data for trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 200:
        return np.zeros(n)
    
    close_1w = df_1w['close'].values
    
    # Calculate 1w EMA200 for trend filter
    ema200_1w = pd.Series(close_1w).ewm(span=200, adjust=False, min_periods=200).mean().values
    ema200_1w_aligned = align_htf_to_ltf(prices, df_1w, ema200_1w)
    
    # Get 1d data for Camarilla pivot levels (prior day)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate Camarilla pivot levels for prior 1d
    camarilla_r4 = np.full(n, np.nan)
    camarilla_s4 = np.full(n, np.nan)
    camarilla_r3 = np.full(n, np.nan)
    camarilla_s3 = np.full(n, np.nan)
    
    # Camarilla formulas based on prior day's range
    for i in range(1, n):
        # Use prior 1d candle (index i-1) to calculate levels for current 6h bar
        prior_high = high_1d[i-1] if i-1 < len(high_1d) else high_1d[-1]
        prior_low = low_1d[i-1] if i-1 < len(low_1d) else low_1d[-1]
        prior_close = close_1d[i-1] if i-1 < len(close_1d) else close_1d[-1]
        
        if prior_high > prior_low:  # valid range
            range_val = prior_high - prior_low
            camarilla_r4[i] = prior_close + range_val * 1.500  # R4 = C + ((H-L) * 1.5)
            camarilla_s4[i] = prior_close - range_val * 1.500  # S4 = C - ((H-L) * 1.5)
            camarilla_r3[i] = prior_close + range_val * 1.250  # R3 = C + ((H-L) * 1.25)
            camarilla_s3[i] = prior_close - range_val * 1.250  # S3 = C - ((H-L) * 1.25)
        else:
            camarilla_r4[i] = prior_close
            camarilla_s4[i] = prior_close
            camarilla_r3[i] = prior_close
            camarilla_s3[i] = prior_close
    
    # Volume confirmation: volume > 2.0 * 20-period average
    vol_ma = np.full(n, np.nan)
    for i in range(20, n):
        vol_ma[i] = np.mean(volume[i-20:i])
    volume_spike = volume > (2.0 * vol_ma)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(50, n):
        # Skip if data not ready
        if (np.isnan(ema200_1w_aligned[i]) or np.isnan(camarilla_r4[i]) or 
            np.isnan(camarilla_s4[i]) or np.isnan(vol_ma[i])):
            signals[i] = 0.0
            continue
        
        # Determine 1w trend
        bullish_trend = close[i] > ema200_1w_aligned[i]
        bearish_trend = close[i] < ema200_1w_aligned[i]
        
        # Entry logic: Camarilla S4/R4 breakout with volume and trend filter
        long_entry = False
        short_entry = False
        
        # Long breakout: price breaks above Camarilla R4 in bullish weekly trend
        if bullish_trend:
            long_entry = (close[i] > camarilla_r4[i]) and volume_spike[i]
        # Short breakout: price breaks below Camarilla S4 in bearish weekly trend
        elif bearish_trend:
            short_entry = (close[i] < camarilla_s4[i]) and volume_spike[i]
        
        # Exit logic: opposite Camarilla level or trend reversal
        long_exit = (bearish_trend and close[i] < camarilla_s3[i]) or (not bullish_trend and not bearish_trend)
        short_exit = (bullish_trend and close[i] > camarilla_r3[i]) or (not bullish_trend and not bearish_trend)
        
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

name = "6h_1w_camarilla_breakout_trend_volume_v1"
timeframe = "6h"
leverage = 1.0
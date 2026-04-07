#!/usr/bin/env python3
"""
1d_camarilla_pivot_1w_volume_v1
Hypothesis: On daily timeframe, use weekly Camarilla pivot levels for mean reversion at S3/R3 and breakout continuation at S4/R4, with weekly EMA50 for trend filter and volume confirmation. Targets 15-25 trades/year to minimize fee drag while capturing both reversals and breakouts. Works in bull (breakouts at S4/R4) and bear (reversals at S3/R3) markets by adapting to price action relative to pivots and trend.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "1d_camarilla_pivot_1w_volume_v1"
timeframe = "1d"
leverage = 1.0

def calculate_camarilla(high, low, close):
    """Calculate Camarilla pivot levels for given high, low, close."""
    range_val = high - low
    if range_val <= 0:
        return close, close, close, close, close, close, close, close
    pivot = (high + low + close) / 3
    s1 = close - (range_val * 1.1 / 12)
    s2 = close - (range_val * 1.1 / 6)
    s3 = close - (range_val * 1.1 / 4)
    s4 = close - (range_val * 1.1 / 2)
    r1 = close + (range_val * 1.1 / 12)
    r2 = close + (range_val * 1.1 / 6)
    r3 = close + (range_val * 1.1 / 4)
    r4 = close + (range_val * 1.1 / 2)
    return pivot, s1, s2, s3, s4, r1, r2, r3, r4

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Price and volume data
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Get weekly data for Camarilla pivots and trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 50:
        return np.zeros(n)
    
    w_high = df_1w['high'].values
    w_low = df_1w['low'].values
    w_close = df_1w['close'].values
    
    # Calculate weekly Camarilla levels
    camarilla_data = np.array([calculate_camarilla(w_high[i], w_low[i], w_close[i]) 
                               for i in range(len(w_close))])
    # Columns: pivot, s1, s2, s3, s4, r1, r2, r3, r4
    camarilla_pivot = camarilla_data[:, 0]
    camarilla_s3 = camarilla_data[:, 3]
    camarilla_s4 = camarilla_data[:, 4]
    camarilla_r3 = camarilla_data[:, 7]
    camarilla_r4 = camarilla_data[:, 8]
    
    # Calculate weekly EMA50 for trend filter
    weekly_close_series = pd.Series(w_close)
    ema50 = weekly_close_series.ewm(span=50, adjust=False).mean().values
    ema50_aligned = align_htf_to_ltf(prices, df_1w, ema50)
    
    # Align Camarilla levels to daily timeframe
    camarilla_s3_aligned = align_htf_to_ltf(prices, df_1w, camarilla_s3)
    camarilla_s4_aligned = align_htf_to_ltf(prices, df_1w, camarilla_s4)
    camarilla_r3_aligned = align_htf_to_ltf(prices, df_1w, camarilla_r3)
    camarilla_r4_aligned = align_htf_to_ltf(prices, df_1w, camarilla_r4)
    
    # Volume filter: daily volume > 1.5x 20-period average
    vol_series = pd.Series(volume)
    vol_ma = vol_series.rolling(window=20, min_periods=20).mean()
    vol_ratio = vol_series / vol_ma
    vol_ratio = vol_ratio.fillna(0).values
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(20, n):  # Start after volume MA warmup
        # Skip if weekly EMA not available
        if np.isnan(ema50_aligned[i]):
            signals[i] = 0.0
            continue
        
        # Skip if Camarilla levels not available
        if np.isnan(camarilla_s3_aligned[i]) or np.isnan(camarilla_r3_aligned[i]):
            signals[i] = 0.0
            continue
        
        # Determine market regime based on price vs EMA50
        uptrend = close[i] > ema50_aligned[i]
        downtrend = close[i] < ema50_aligned[i]
        
        # Volume confirmation
        vol_confirmed = vol_ratio[i] > 1.5
        
        if position == 1:  # Long position
            # Exit conditions
            exit_long = False
            # Exit when price breaks below S3 (mean reversion failure)
            if close[i] < camarilla_s3_aligned[i]:
                exit_long = True
            # Exit when trend turns down
            elif not uptrend:
                exit_long = True
            # Exit when volume drops significantly
            elif vol_ratio[i] < 0.8:
                exit_long = True
            
            if exit_long:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit conditions
            exit_short = False
            # Exit when price breaks above R3 (mean reversion failure)
            if close[i] > camarilla_r3_aligned[i]:
                exit_short = True
            # Exit when trend turns up
            elif not downtrend:
                exit_short = True
            # Exit when volume drops significantly
            elif vol_ratio[i] < 0.8:
                exit_short = True
            
            if exit_short:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat, look for entry
            # Long entry at S3 (mean reversion) OR breakout above S4 with trend
            long_s3 = (close[i] <= camarilla_s3_aligned[i]) and uptrend and vol_confirmed
            long_s4 = (close[i] > camarilla_s4_aligned[i]) and uptrend and vol_confirmed
            
            # Short entry at R3 (mean reversion) OR breakdown below R4 with trend
            short_r3 = (close[i] >= camarilla_r3_aligned[i]) and downtrend and vol_confirmed
            short_r4 = (close[i] < camarilla_r4_aligned[i]) and downtrend and vol_confirmed
            
            if long_s3 or long_s4:
                position = 1
                signals[i] = 0.25
            elif short_r3 or short_r4:
                position = -1
                signals[i] = -0.25
    
    return signals
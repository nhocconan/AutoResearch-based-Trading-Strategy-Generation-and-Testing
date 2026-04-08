#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Camarilla Pivot Breakout with 1w Trend Filter
# Uses daily Camarilla pivot levels (R3/S3 for mean reversion, R4/S4 for breakout).
# Trades breakouts in the direction of the weekly trend (EMA20 slope).
# Mean reversion at R3/S3 when weekly trend is weak (EMA20 flat).
# Works in bull/bear by adapting to weekly trend direction.
# Target: 12-37 trades/year via strict pivot level + trend confluence.
name = "6h_camarilla_pivot_1w_trend_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Price data
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    
    # Get daily data for Camarilla pivots (call ONCE before loop)
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate Camarilla pivot levels for each daily bar
    # Formula: 
    # PP = (H + L + C) / 3
    # R4 = PP + (H - L) * 1.1/2
    # R3 = PP + (H - L) * 1.1/4
    # S3 = PP - (H - L) * 1.1/4
    # S4 = PP - (H - L) * 1.1/2
    pp = (high_1d + low_1d + close_1d) / 3
    r4 = pp + (high_1d - low_1d) * 1.1 / 2
    r3 = pp + (high_1d - low_1d) * 1.1 / 4
    s3 = pp - (high_1d - low_1d) * 1.1 / 4
    s4 = pp - (high_1d - low_1d) * 1.1 / 2
    
    # Get weekly data for EMA20 trend filter (call ONCE before loop)
    df_1w = get_htf_data(prices, '1w')
    close_1w = df_1w['close'].values
    
    # Calculate EMA20 on weekly close
    ema_20 = pd.Series(close_1w).ewm(span=20, adjust=False).mean().values
    
    # Calculate EMA20 slope for trend strength
    ema_slope = np.diff(ema_20, prepend=ema_20[0])
    
    # Align all HTF data to 6s timeframe
    r3_6h = align_htf_to_ltf(prices, df_1d, r3)
    r4_6h = align_htf_to_ltf(prices, df_1d, r4)
    s3_6h = align_htf_to_ltf(prices, df_1d, s3)
    s4_6h = align_htf_to_ltf(prices, df_1d, s4)
    ema_slope_6h = align_htf_to_ltf(prices, df_1w, ema_slope)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    # Start from sufficient lookback
    start_idx = 10
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(r3_6h[i]) or np.isnan(r4_6h[i]) or 
            np.isnan(s3_6h[i]) or np.isnan(s4_6h[i]) or 
            np.isnan(ema_slope_6h[i])):
            signals[i] = 0.0
            continue
        
        if position == 1:  # Long position
            # Exit: price closes below S3 OR weekly trend turns negative
            if close[i] < s3_6h[i] or ema_slope_6h[i] < 0:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit: price closes above R3 OR weekly trend turns positive
            if close[i] > r3_6h[i] or ema_slope_6h[i] > 0:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat, look for entry
            # Breakout mode: strong weekly trend (|slope| > 0.1% of price)
            strong_trend = abs(ema_slope_6h[i]) > 0.001 * close[i]
            
            if strong_trend:
                # Breakout continuation: trade in trend direction
                if ema_slope_6h[i] > 0 and close[i] > r4_6h[i]:  # Uptrend + break above R4
                    position = 1
                    signals[i] = 0.25
                elif ema_slope_6h[i] < 0 and close[i] < s4_6h[i]:  # Downtrend + break below S4
                    position = -1
                    signals[i] = -0.25
            else:
                # Mean reversion mode: weak trend, fade at R3/S3
                if close[i] > r3_6h[i]:  # Sell at R3 resistance
                    position = -1
                    signals[i] = -0.25
                elif close[i] < s3_6h[i]:  # Buy at S3 support
                    position = 1
                    signals[i] = 0.25
    
    return signals
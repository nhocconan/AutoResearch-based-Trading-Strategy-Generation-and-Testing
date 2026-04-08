#!/usr/bin/env python3
"""
6h_camarilla_pivot_12h_trend_volume_v1
Hypothesis: On 6h timeframe, use daily Camarilla pivot levels for mean reversion at R3/S3 and breakout continuation at R4/S4, with 12h EMA trend filter and volume confirmation. In ranging markets, fade extremes at R3/S3; in trending markets (price > 12h EMA50), breakout continuation at R4/S4. Volume confirmation ensures institutional participation. Designed for low trade frequency to minimize fee impact while capturing both mean reversion and trend continuation opportunities.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "6h_camarilla_pivot_12h_trend_volume_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 60:
        return np.zeros(n)
    
    # Price data
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Daily data for Camarilla pivots
    df_daily = get_htf_data(prices, '1d')
    if len(df_daily) < 30:
        return np.zeros(n)
    
    # Calculate daily Camarilla pivot levels
    daily_high = df_daily['high'].values
    daily_low = df_daily['low'].values
    daily_close = df_daily['close'].values
    
    # Camarilla calculations
    daily_range = daily_high - daily_low
    camarilla_pivot = (daily_high + daily_low + daily_close * 2) / 4
    camarilla_r3 = camarilla_pivot + 1.1 * daily_range / 2
    camarilla_s3 = camarilla_pivot - 1.1 * daily_range / 2
    camarilla_r4 = camarilla_pivot + 1.1 * daily_range
    camarilla_s4 = camarilla_pivot - 1.1 * daily_range
    
    # Align Camarilla levels to 6h timeframe
    camarilla_pivot_6h = align_htf_to_ltf(prices, df_daily, camarilla_pivot)
    camarilla_r3_6h = align_htf_to_ltf(prices, df_daily, camarilla_r3)
    camarilla_s3_6h = align_htf_to_ltf(prices, df_daily, camarilla_s3)
    camarilla_r4_6h = align_htf_to_ltf(prices, df_daily, camarilla_r4)
    camarilla_s4_6h = align_htf_to_ltf(prices, df_daily, camarilla_s4)
    
    # 12h EMA50 for trend filter
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 50:
        return np.zeros(n)
    
    ema_12h_50 = df_12h['close'].ewm(span=50, adjust=False).mean().values
    ema_12h_50_6h = align_htf_to_ltf(prices, df_12h, ema_12h_50)
    
    # Volume confirmation (24-period average on 6h = 6 days)
    vol_ma = pd.Series(volume).rolling(window=24, min_periods=24).mean().values
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(24, n):
        # Skip if required data not available
        if (np.isnan(camarilla_r3_6h[i]) or np.isnan(camarilla_s3_6h[i]) or
            np.isnan(camarilla_r4_6h[i]) or np.isnan(camarilla_s4_6h[i]) or
            np.isnan(ema_12h_50_6h[i]) or np.isnan(vol_ma[i]) or vol_ma[i] <= 0):
            signals[i] = 0.0
            continue
        
        # Volume confirmation: current volume > 1.3x 24-period average
        vol_confirm = volume[i] > 1.3 * vol_ma[i]
        
        # Trend determination: price vs 12h EMA50
        uptrend = close[i] > ema_12h_50_6h[i]
        downtrend = close[i] < ema_12h_50_6h[i]
        
        if position == 1:  # Long position
            # Exit conditions
            exit_long = False
            # Exit if price crosses below Camarilla S3 in ranging market
            if not uptrend and close[i] < camarilla_s3_6h[i]:
                exit_long = True
            # Exit if price crosses below Camarilla S4 (strong reversal)
            elif close[i] < camarilla_s4_6h[i]:
                exit_long = True
            # Exit if trend turns down and we're at resistance
            elif downtrend and close[i] > camarilla_r3_6h[i]:
                exit_long = True
            
            if exit_long:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit conditions
            exit_short = False
            # Exit if price crosses above Camarilla R3 in ranging market
            if not downtrend and close[i] > camarilla_r3_6h[i]:
                exit_short = True
            # Exit if price crosses above Camarilla R4 (strong reversal)
            elif close[i] > camarilla_r4_6h[i]:
                exit_short = True
            # Exit if trend turns up and we're at support
            elif uptrend and close[i] < camarilla_s3_6h[i]:
                exit_short = True
            
            if exit_short:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat, look for entry
            # Long entry conditions
            long_entry = False
            # Mean reversion long: price at S3 support in ranging/downtrend
            if (not uptrend or downtrend) and camarilla_s3_6h[i] <= close[i] <= camarilla_s3_6h[i] * 1.002:
                if vol_confirm:
                    long_entry = True
            # Breakout continuation long: price breaks R4 in uptrend
            elif uptrend and close[i] > camarilla_r4_6h[i]:
                if vol_confirm:
                    long_entry = True
            
            # Short entry conditions
            short_entry = False
            # Mean reversion short: price at R3 resistance in ranging/uptrend
            if (not downtrend or uptrend) and camarilla_r3_6h[i] * 0.998 <= close[i] <= camarilla_r3_6h[i]:
                if vol_confirm:
                    short_entry = True
            # Breakout continuation short: price breaks S4 in downtrend
            elif downtrend and close[i] < camarilla_s4_6h[i]:
                if vol_confirm:
                    short_entry = True
            
            if long_entry:
                position = 1
                signals[i] = 0.25
            elif short_entry:
                position = -1
                signals[i] = -0.25
    
    return signals
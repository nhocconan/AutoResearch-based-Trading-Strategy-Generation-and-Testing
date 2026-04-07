#!/usr/bin/env python3
"""
4h_camarilla_pivot_1d_trend_volume_v2
Hypothesis: On 4h timeframe, use daily Camarilla pivot levels for mean reversion at R3/S3 and breakout continuation at R4/S4, with 1d EMA trend filter and volume confirmation. In ranging markets (price < 1d EMA200), fade extremes at R3/S3; in trending markets (price > 1d EMA200), breakout continuation at R4/S4. Volume confirmation ensures institutional participation. Designed for low trade frequency to minimize fee impact while capturing both mean reversion and trend continuation opportunities. Works in bull/bear via trend filter adapting to regime.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "4h_camarilla_pivot_1d_trend_volume_v2"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Price data
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Daily data for Camarilla pivots and trend filter
    df_daily = get_htf_data(prices, '1d')
    if len(df_daily) < 50:
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
    
    # Align Camarilla levels to 4h timeframe
    camarilla_pivot_4h = align_htf_to_ltf(prices, df_daily, camarilla_pivot)
    camarilla_r3_4h = align_htf_to_ltf(prices, df_daily, camarilla_r3)
    camarilla_s3_4h = align_htf_to_ltf(prices, df_daily, camarilla_s3)
    camarilla_r4_4h = align_htf_to_ltf(prices, df_daily, camarilla_r4)
    camarilla_s4_4h = align_htf_to_ltf(prices, df_daily, camarilla_s4)
    
    # Daily EMA200 for trend filter
    ema_daily_200 = pd.Series(df_daily['close'].values).ewm(span=200, adjust=False).mean().values
    ema_daily_200_4h = align_htf_to_ltf(prices, df_daily, ema_daily_200)
    
    # Volume confirmation (20-period average on 4h = ~3.3 days)
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(20, n):
        # Skip if required data not available
        if (np.isnan(camarilla_r3_4h[i]) or np.isnan(camarilla_s3_4h[i]) or
            np.isnan(camarilla_r4_4h[i]) or np.isnan(camarilla_s4_4h[i]) or
            np.isnan(ema_daily_200_4h[i]) or np.isnan(vol_ma[i]) or vol_ma[i] <= 0):
            signals[i] = 0.0
            continue
        
        # Volume confirmation: current volume > 1.5x 20-period average
        vol_confirm = volume[i] > 1.5 * vol_ma[i]
        
        # Trend determination: price vs daily EMA200
        uptrend = close[i] > ema_daily_200_4h[i]
        downtrend = close[i] < ema_daily_200_4h[i]
        
        if position == 1:  # Long position
            # Exit conditions
            exit_long = False
            # Exit if price crosses below Camarilla S3 in ranging market
            if not uptrend and close[i] < camarilla_s3_4h[i]:
                exit_long = True
            # Exit if price crosses below Camarilla S4 (strong reversal)
            elif close[i] < camarilla_s4_4h[i]:
                exit_long = True
            # Exit if trend turns down and we're at resistance
            elif downtrend and close[i] > camarilla_r3_4h[i]:
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
            if not downtrend and close[i] > camarilla_r3_4h[i]:
                exit_short = True
            # Exit if price crosses above Camarilla R4 (strong reversal)
            elif close[i] > camarilla_r4_4h[i]:
                exit_short = True
            # Exit if trend turns up and we're at support
            elif uptrend and close[i] < camarilla_s3_4h[i]:
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
            if (not uptrend or downtrend) and camarilla_s3_4h[i] <= close[i] <= camarilla_s3_4h[i] * 1.002:
                if vol_confirm:
                    long_entry = True
            # Breakout continuation long: price breaks R4 in uptrend
            elif uptrend and close[i] > camarilla_r4_4h[i]:
                if vol_confirm:
                    long_entry = True
            
            # Short entry conditions
            short_entry = False
            # Mean reversion short: price at R3 resistance in ranging/uptrend
            if (not downtrend or uptrend) and camarilla_r3_4h[i] * 0.998 <= close[i] <= camarilla_r3_4h[i]:
                if vol_confirm:
                    short_entry = True
            # Breakout continuation short: price breaks S4 in downtrend
            elif downtrend and close[i] < camarilla_s4_4h[i]:
                if vol_confirm:
                    short_entry = True
            
            if long_entry:
                position = 1
                signals[i] = 0.25
            elif short_entry:
                position = -1
                signals[i] = -0.25
    
    return signals
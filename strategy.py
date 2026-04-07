#!/usr/bin/env python3
"""
12h_camarilla_pivot_1d_volume_v1
Hypothesis: On 12-hour timeframe, use Camarilla pivot levels derived from daily candles.
Long when price approaches S3 support with volume spike and daily trend up.
Short when price approaches R3 resistance with volume spike and daily trend down.
Exit when price reaches opposite pivot level (S1/R1).
Designed for 15-25 trades/year to minimize fee decay while capturing institutional reversal points.
Works in both bull/bear markets as Camarilla adapts to volatility and daily trend filter avoids counter-trend trades.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "12h_camarilla_pivot_1d_volume_v1"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Price data
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Get 1d data for Camarilla pivots and trend filter
    df_1d = get_htf_data(prices, '1d')
    
    if len(df_1d) < 20:
        return np.zeros(n)
    
    # Calculate daily OHLC
    daily_high = df_1d['high'].values
    daily_low = df_1d['low'].values
    daily_close = df_1d['close'].values
    
    # Calculate Camarilla levels for each day
    # R4 = Close + ((High - Low) * 1.1/2)
    # R3 = Close + ((High - Low) * 1.1/4)
    # R2 = Close + ((High - Low) * 1.1/6)
    # R1 = Close + ((High - Low) * 1.1/12)
    # S1 = Close - ((High - Low) * 1.1/12)
    # S2 = Close - ((High - Low) * 1.1/6)
    # S3 = Close - ((High - Low) * 1.1/4)
    # S4 = Close - ((High - Low) * 1.1/2)
    
    camarilla_r3 = daily_close + ((daily_high - daily_low) * 1.1 / 4)
    camarilla_s3 = daily_close - ((daily_high - daily_low) * 1.1 / 4)
    camarilla_r1 = daily_close + ((daily_high - daily_low) * 1.1 / 12)
    camarilla_s1 = daily_close - ((daily_high - daily_low) * 1.1 / 12)
    
    # Align Camarilla levels to 12h timeframe
    r3_12h = align_htf_to_ltf(prices, df_1d, camarilla_r3)
    s3_12h = align_htf_to_ltf(prices, df_1d, camarilla_s3)
    r1_12h = align_htf_to_ltf(prices, df_1d, camarilla_r1)
    s1_12h = align_htf_to_ltf(prices, df_1d, camarilla_s1)
    
    # Daily trend filter using EMA(20)
    daily_ema20 = pd.Series(daily_close).ewm(span=20, adjust=False, min_periods=20).mean().values
    daily_ema20_aligned = align_htf_to_ltf(prices, df_1d, daily_ema20)
    
    # Determine daily trend direction (using price vs EMA)
    daily_trend_up = daily_close > daily_ema20
    daily_trend_down = daily_close < daily_ema20
    daily_trend_up_aligned = align_htf_to_ltf(prices, df_1d, daily_trend_up)
    daily_trend_down_aligned = align_htf_to_ltf(prices, df_1d, daily_trend_down)
    
    # Volume filter: 20-period average on 12h timeframe
    vol_series = pd.Series(volume)
    vol_ma = vol_series.rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(max(20, 50), n):
        # Skip if data not available
        if (np.isnan(r3_12h[i]) or np.isnan(s3_12h[i]) or 
            np.isnan(r1_12h[i]) or np.isnan(s1_12h[i]) or 
            np.isnan(daily_ema20_aligned[i]) or np.isnan(vol_ma[i])):
            signals[i] = 0.0
            continue
            
        # Volume confirmation
        vol_ok = volume[i] > 1.5 * vol_ma[i]
        
        if position == 1:  # Long position
            # Exit: price reaches S1 (support 1)
            if close[i] <= s1_12h[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit: price reaches R1 (resistance 1)
            if close[i] >= r1_12h[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat, look for entry
            # Only enter with volume confirmation and daily trend alignment
            if vol_ok:
                # Long: price near S3 with daily uptrend
                if (close[i] <= s3_12h[i] * 1.005 and  # within 0.5% of S3
                    daily_trend_up_aligned[i]):
                    position = 1
                    signals[i] = 0.25
                # Short: price near R3 with daily downtrend
                elif (close[i] >= r3_12h[i] * 0.995 and  # within 0.5% of R3
                      daily_trend_down_aligned[i]):
                    position = -1
                    signals[i] = -0.25
    
    return signals
#!/usr/bin/env python3
"""
6h_camarilla_pivot_1w_trend_volume_v1
Hypothesis: Camarilla pivot levels from weekly timeframe provide strong support/resistance.
Buy at S3/S4 level with bullish weekly trend and volume confirmation.
Sell at R3/R4 level with bearish weekly trend and volume confirmation.
Weekly trend filter prevents counter-trend trading in strong moves.
Designed for 15-25 trades/year on 6h timeframe with clear reversal/continuation logic.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "6h_camarilla_pivot_1w_trend_volume_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Price and volume data
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Weekly data for Camarilla pivots and trend
    df_weekly = get_htf_data(prices, '1w')
    if len(df_weekly) < 50:
        return np.zeros(n)
    
    # Calculate weekly Camarilla levels
    weekly_high = df_weekly['high'].values
    weekly_low = df_weekly['low'].values
    weekly_close = df_weekly['close'].values
    
    # Camarilla formula: range = H - L
    # Resistance levels: R3 = C + (H-L)*1.1/2, R4 = C + (H-L)*1.1
    # Support levels: S3 = C - (H-L)*1.1/2, S4 = C - (H-L)*1.1
    weekly_range = weekly_high - weekly_low
    r3 = weekly_close + weekly_range * 1.1 / 2
    r4 = weekly_close + weekly_range * 1.1
    s3 = weekly_close - weekly_range * 1.1 / 2
    s4 = weekly_close - weekly_range * 1.1
    
    # Weekly EMA50 for trend filter
    ema50_weekly = pd.Series(weekly_close).ewm(span=50, adjust=False).mean().values
    weekly_uptrend = weekly_close > ema50_weekly
    weekly_downtrend = weekly_close < ema50_weekly
    
    # Align weekly data to 6h timeframe (shifted by 1 for completed weekly bars only)
    r3_6h = align_htf_to_ltf(prices, df_weekly, r3)
    r4_6h = align_htf_to_ltf(prices, df_weekly, r4)
    s3_6h = align_htf_to_ltf(prices, df_weekly, s3)
    s4_6h = align_htf_to_ltf(prices, df_weekly, s4)
    weekly_uptrend_6h = align_htf_to_ltf(prices, df_weekly, weekly_uptrend.astype(float))
    weekly_downtrend_6h = align_htf_to_ltf(prices, df_weekly, weekly_downtrend.astype(float))
    
    # Volume confirmation: 24-period average (4 days of 6h bars)
    vol_ma = pd.Series(volume).rolling(window=24, min_periods=24).mean().values
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(24, n):
        # Skip if data not available
        if (np.isnan(r3_6h[i]) or np.isnan(r4_6h[i]) or np.isnan(s3_6h[i]) or np.isnan(s4_6h[i]) or
            np.isnan(weekly_uptrend_6h[i]) or np.isnan(weekly_downtrend_6h[i]) or
            np.isnan(vol_ma[i]) or vol_ma[i] == 0):
            signals[i] = 0.0
            continue
        
        # Volume confirmation: current volume above average
        vol_confirmed = volume[i] > vol_ma[i]
        
        if position == 1:  # Long position
            # Exit: price reaches R3/R4 or weekly trend turns bearish
            if close[i] >= r3_6h[i] or weekly_downtrend_6h[i] > 0.5:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit: price reaches S3/S4 or weekly trend turns bullish
            if close[i] <= s3_6h[i] or weekly_uptrend_6h[i] > 0.5:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat, look for entry
            # Long: price at S3/S4 level with bullish weekly trend and volume confirmation
            if vol_confirmed and weekly_uptrend_6h[i] > 0.5:
                if close[i] <= s3_6h[i] * 1.005:  # Within 0.5% of S3
                    position = 1
                    signals[i] = 0.25
                elif close[i] <= s4_6h[i] * 1.005:  # Within 0.5% of S4
                    position = 1
                    signals[i] = 0.25
            
            # Short: price at R3/R4 level with bearish weekly trend and volume confirmation
            elif vol_confirmed and weekly_downtrend_6h[i] > 0.5:
                if close[i] >= r3_6h[i] * 0.995:  # Within 0.5% of R3
                    position = -1
                    signals[i] = -0.25
                elif close[i] >= r4_6h[i] * 0.995:  # Within 0.5% of R4
                    position = -1
                    signals[i] = -0.25
    
    return signals
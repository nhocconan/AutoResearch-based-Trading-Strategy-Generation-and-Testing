# 6h Weekly Pivot Reversal with Daily Trend Filter
# Hypothesis: Weekly pivot points identify key institutional levels. Price often reverses
# from weekly R1/S1 or breaks through R2/S2 with strong daily trend (ADX>25). 
# Works in both bull/bear markets by fading at inner pivots and breaking outer pivots.
# Target: 15-25 trades/year per symbol.

name = "6h_weekly_pivot_reversal_daily_trend_v1"
timeframe = "6h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Price data
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    
    # Get weekly data for pivot points - call ONCE before loop
    df_w = get_htf_data(prices, '1w')
    high_w = df_w['high'].values
    low_w = df_w['low'].values
    close_w = df_w['close'].values
    
    # Get daily data for trend filter - call ONCE before loop
    df_d = get_htf_data(prices, '1d')
    high_d = df_d['high'].values
    low_d = df_d['low'].values
    close_d = df_d['close'].values
    
    # Calculate weekly pivot points (standard floor trader pivots)
    # P = (H + L + C) / 3
    # R1 = 2*P - L, S1 = 2*P - H
    # R2 = P + (H - L), S2 = P - (H - L)
    # R3 = H + 2*(P - L), S3 = L - 2*(H - P)
    pp_w = (high_w + low_w + close_w) / 3
    r1_w = 2 * pp_w - low_w
    s1_w = 2 * pp_w - high_w
    r2_w = pp_w + (high_w - low_w)
    s2_w = pp_w - (high_w - low_w)
    r3_w = high_w + 2 * (pp_w - low_w)
    s3_w = low_w - 2 * (high_w - pp_w)
    
    # Calculate 14-period ADX for daily trend filter
    # True Range
    tr1 = high_d[1:] - low_d[1:]
    tr2 = np.abs(high_d[1:] - close_d[:-1])
    tr3 = np.abs(low_d[1:] - close_d[:-1])
    tr = np.concatenate([[np.nan], np.maximum(tr1, np.maximum(tr2, tr3))])
    
    # Directional Movement
    dm_plus = np.where((high_d[1:] - high_d[:-1]) > (low_d[:-1] - low_d[1:]), 
                       np.maximum(high_d[1:] - high_d[:-1], 0), 0)
    dm_minus = np.where((low_d[:-1] - low_d[1:]) > (high_d[1:] - high_d[:-1]), 
                        np.maximum(low_d[:-1] - low_d[1:], 0), 0)
    dm_plus = np.concatenate([[0], dm_plus])
    dm_minus = np.concatenate([[0], dm_minus])
    
    # Smoothed values
    tr14 = pd.Series(tr).rolling(window=14, min_periods=14).sum().values
    dm_plus_14 = pd.Series(dm_plus).rolling(window=14, min_periods=14).sum().values
    dm_minus_14 = pd.Series(dm_minus).rolling(window=14, min_periods=14).sum().values
    
    # Directional Indicators
    di_plus = 100 * dm_plus_14 / tr14
    di_minus = 100 * dm_minus_14 / tr14
    
    # DX and ADX
    dx = 100 * np.abs(di_plus - di_minus) / (di_plus + di_minus)
    adx_d = pd.Series(dx).rolling(window=14, min_periods=14).mean().values
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    # Start from sufficient lookback
    start_idx = 30
    
    for i in range(start_idx, n):
        # Get aligned weekly pivot values for current 6h bar
        pp = align_htf_to_ltf(prices, df_w, pp_w)[i]
        r1 = align_htf_to_ltf(prices, df_w, r1_w)[i]
        s1 = align_htf_to_ltf(prices, df_w, s1_w)[i]
        r2 = align_htf_to_ltf(prices, df_w, r2_w)[i]
        s2 = align_htf_to_ltf(prices, df_w, s2_w)[i]
        r3 = align_htf_to_ltf(prices, df_w, r3_w)[i]
        s3 = align_htf_to_ltf(prices, df_w, s3_w)[i]
        
        # Get aligned daily ADX for current 6h bar
        adx = align_htf_to_ltf(prices, df_d, adx_d)[i]
        
        # Skip if any required data is NaN
        if np.isnan(pp) or np.isnan(r1) or np.isnan(s1) or np.isnan(adx):
            signals[i] = 0.0
            continue
        
        # Strong daily trend filter
        strong_trend = adx > 25
        
        if position == 1:  # Long position
            # Exit if price breaks below S1 in strong trend (trend reversal)
            # OR if price reaches R3 (take profit at strong resistance)
            if strong_trend and (close[i] < s1 or close[i] > r3):
                position = 0
                signals[i] = 0.0
            elif position == 1:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit if price breaks above R1 in strong trend (trend reversal)
            # OR if price reaches S3 (take profit at strong support)
            if strong_trend and (close[i] > r1 or close[i] < s3):
                position = 0
                signals[i] = 0.0
            elif position == -1:
                signals[i] = -0.25
        else:  # Flat, look for entry
            # Fade at inner pivots (R1/S1) in strong trend
            # Breakout at outer pivots (R2/S2) with strong trend confirmation
            if strong_trend:
                # Fade long at S1 (support) - price approaching support in uptrend
                if close[i] <= s1 * 1.002 and close[i-1] > s1 * 1.002:  # slight buffer
                    position = 1
                    signals[i] = 0.25
                # Fade short at R1 (resistance) - price approaching resistance in downtrend
                elif close[i] >= r1 * 0.998 and close[i-1] < r1 * 0.998:  # slight buffer
                    position = -1
                    signals[i] = -0.25
                # Breakout long above R2 with strong uptrend
                elif close[i] >= r2 * 0.998 and close[i-1] < r2 * 0.998:
                    position = 1
                    signals[i] = 0.25
                # Breakout short below S2 with strong downtrend
                elif close[i] <= s2 * 1.002 and close[i-1] > s2 * 1.002:
                    position = -1
                    signals[i] = -0.25
    
    return signals
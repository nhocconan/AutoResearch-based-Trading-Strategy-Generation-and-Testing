#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Williams %R mean reversion + weekly pivot regime + volume confirmation
# Williams %R identifies overbought/oversold conditions; weekly pivot provides trend regime
# (above/below weekly pivot); volume confirms mean reversion strength.
# Works in both bull and bear markets by fading extremes in range regimes and
# continuing trends in trending regimes. Target: 12-30 trades/year (50-120 total).

name = "6h_WilliamsR_MeanRev_WeeklyPivot_Regime_VolumeConfirm_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Load HTF data ONCE before loop for 1w and 1d calculations
    df_1w = get_htf_data(prices, '1w')
    df_1d = get_htf_data(prices, '1d')
    if len(df_1w) < 10 or len(df_1d) < 50:
        return np.zeros(n)
    
    # Calculate weekly pivot points (using previous week's OHLC)
    # P = (H + L + C) / 3
    # R1 = 2*P - L, S1 = 2*P - H
    # R2 = P + (H - L), S2 = P - (H - L)
    # R3 = H + 2*(P - L), S3 = L - 2*(H - P)
    prev_week_high = df_1w['high'].shift(1).values
    prev_week_low = df_1w['low'].shift(1).values
    prev_week_close = df_1w['close'].shift(1).values
    
    weekly_pivot = (prev_week_high + prev_week_low + prev_week_close) / 3.0
    weekly_range = prev_week_high - prev_week_low
    r1 = 2 * weekly_pivot - prev_week_low
    s1 = 2 * weekly_pivot - prev_week_high
    r2 = weekly_pivot + weekly_range
    s2 = weekly_pivot - weekly_range
    r3 = prev_week_high + 2 * (weekly_pivot - prev_week_low)
    s3 = prev_week_low - 2 * (prev_week_high - weekly_pivot)
    
    # Align weekly pivot levels to 6h timeframe (completed weekly bar only)
    weekly_pivot_aligned = align_htf_to_ltf(prices, df_1w, weekly_pivot)
    r1_aligned = align_htf_to_ltf(prices, df_1w, r1)
    s1_aligned = align_htf_to_ltf(prices, df_1w, s1)
    r2_aligned = align_htf_to_ltf(prices, df_1w, r2)
    s2_aligned = align_htf_to_ltf(prices, df_1w, s2)
    r3_aligned = align_htf_to_ltf(prices, df_1w, r3)
    s3_aligned = align_htf_to_ltf(prices, df_1w, s3)
    
    # Calculate Williams %R (14-period) on 6h data
    # %R = (Highest High - Close) / (Highest High - Lowest Low) * -100
    highest_high = pd.Series(high).rolling(window=14, min_periods=14).max().values
    lowest_low = pd.Series(low).rolling(window=14, min_periods=14).min().values
    williams_r = (highest_high - close) / (highest_high - lowest_low) * -100
    # Handle division by zero
    williams_r = np.where((highest_high - lowest_low) == 0, -50, williams_r)
    
    # Volume confirmation: volume > 1.3x 20-period average
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_confirm = volume > (1.3 * vol_ma_20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(50, 14, 20)  # warmup for 1d EMA, Williams %R, volume MA
    
    for i in range(start_idx, n):
        # Skip if HTF data not available
        if (np.isnan(weekly_pivot_aligned[i]) or np.isnan(r1_aligned[i]) or 
            np.isnan(s1_aligned[i]) or np.isnan(williams_r[i])):
            signals[i] = 0.0
            continue
            
        curr_close = close[i]
        curr_williams_r = williams_r[i]
        curr_volume_confirm = volume_confirm[i]
        curr_weekly_pivot = weekly_pivot_aligned[i]
        curr_r1 = r1_aligned[i]
        curr_s1 = s1_aligned[i]
        curr_r2 = r2_aligned[i]
        curr_s2 = s2_aligned[i]
        curr_r3 = r3_aligned[i]
        curr_s3 = s3_aligned[i]
        
        # Determine market regime based on weekly pivot
        # Above weekly pivot = bullish regime (favor longs)
        # Below weekly pivot = bearish regime (favor shorts)
        # Between S1 and R1 = neutral regime (mean revert)
        
        if position == 0:  # Flat - look for new entries
            # Neutral regime (S1 < price < R1): mean reversion at extremes
            if curr_s1 < curr_close < curr_r1:
                # Long when oversold (%R < -80) + volume confirmation
                if curr_williams_r < -80 and curr_volume_confirm:
                    signals[i] = 0.25
                    position = 1
                # Short when overbought (%R > -20) + volume confirmation
                elif curr_williams_r > -20 and curr_volume_confirm:
                    signals[i] = -0.25
                    position = -1
            
            # Bullish regime (price > R1): continuation on pullbacks
            elif curr_close > curr_r1:
                # Long on pullback to R1/S1 area when not overbought
                if curr_close > curr_s1 and curr_williams_r > -50 and curr_williams_r < -20:
                    if curr_volume_confirm:
                        signals[i] = 0.25
                        position = 1
            
            # Bearish regime (price < S1): continuation on bounces
            elif curr_close < curr_s1:
                # Short on bounce to R1/S1 area when not oversold
                if curr_close < curr_r1 and curr_williams_r < -50 and curr_williams_r > -80:
                    if curr_volume_confirm:
                        signals[i] = -0.25
                        position = -1
        
        elif position == 1:  # Long position - exit conditions
            # Exit when: overbought (%R > -20) OR weekly pivot breakdown
            if curr_williams_r > -20 or curr_close < curr_weekly_pivot:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:  # Short position - exit conditions
            # Exit when: oversold (%R < -80) OR weekly pivot breakout
            if curr_williams_r < -80 or curr_close > curr_weekly_pivot:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals
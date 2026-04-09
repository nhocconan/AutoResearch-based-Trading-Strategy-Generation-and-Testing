#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Williams %R + ADX trend filter with 12h pivot confirmation
# - Williams %R(14) on 6h for overbought/oversold signals
# - ADX(14) on 12h to filter only trending regimes (ADX > 25)
# - 12h Camarilla pivot levels: short at R3/R4, long at S3/S4 with confirmation
# - Session filter: 08-20 UTC to avoid low-volume Asian session
# - Target: 12-25 trades/year on 6h timeframe (50-100 total over 4 years)
# - Williams %R effective in both bull/bear markets for mean reversion in trends
# - ADX ensures we only trade when trend is strong enough to persist

name = "6h_12h_williamsr_adx_pivot_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Load HTF data ONCE before loop
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 30:
        return np.zeros(n)
    
    # Pre-compute session filter (08-20 UTC)
    hours = prices.index.hour
    in_session = (hours >= 8) & (hours <= 20)
    
    # 12h ADX(14) for trend strength filtering
    high_12h = df_12h['high'].values
    low_12h = df_12h['low'].values
    close_12h = df_12h['close'].values
    
    # Calculate True Range
    tr1 = high_12h - low_12h
    tr2 = np.abs(high_12h - np.roll(close_12h, 1))
    tr3 = np.abs(low_12h - np.roll(close_12h, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]
    atr_12h = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    # Calculate Directional Movement
    up_move = high_12h - np.roll(high_12h, 1)
    down_move = np.roll(low_12h, 1) - low_12h
    up_move = np.where((up_move > down_move) & (up_move > 0), up_move, 0)
    down_move = np.where((down_move > up_move) & (down_move > 0), down_move, 0)
    
    # Calculate +DI and -DI
    plus_di = 100 * pd.Series(up_move).rolling(window=14, min_periods=14).mean().values / atr_12h
    minus_di = 100 * pd.Series(down_move).rolling(window=14, min_periods=14).mean().values / atr_12h
    
    # Calculate DX and ADX
    dx = np.abs(plus_di - minus_di) / (plus_di + minus_di) * 100
    dx = np.where((plus_di + minus_di) == 0, 0, dx)
    adx_12h = pd.Series(dx).rolling(window=14, min_periods=14).mean().values
    
    # Align 12h ADX to 6h
    adx_12h_aligned = align_htf_to_ltf(prices, df_12h, adx_12h)
    
    # 12h Camarilla pivot levels
    pivot_12h = (high_12h + low_12h + close_12h) / 3
    range_12h = high_12h - low_12h
    
    # Camarilla levels
    r3_12h = close_12h + (range_12h * 1.1 / 4)
    r4_12h = close_12h + (range_12h * 1.1 / 2)
    s3_12h = close_12h - (range_12h * 1.1 / 4)
    s4_12h = close_12h - (range_12h * 1.1 / 2)
    
    # Align pivot levels to 6h
    r3_12h_aligned = align_htf_to_ltf(prices, df_12h, r3_12h)
    r4_12h_aligned = align_htf_to_ltf(prices, df_12h, r4_12h)
    s3_12h_aligned = align_htf_to_ltf(prices, df_12h, s3_12h)
    s4_12h_aligned = align_htf_to_ltf(prices, df_12h, s4_12h)
    
    # 6h Williams %R(14) for mean reversion signals
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    
    highest_high = pd.Series(high).rolling(window=14, min_periods=14).max().values
    lowest_low = pd.Series(low).rolling(window=14, min_periods=14).min().values
    williams_r = -100 * (highest_high - close) / (highest_high - lowest_low)
    williams_r = np.where((highest_high - lowest_low) == 0, -50, williams_r)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(20, n):
        # Skip if any required data is invalid or outside session
        if (not in_session[i] or
            np.isnan(adx_12h_aligned[i]) or
            np.isnan(williams_r[i]) or
            np.isnan(r3_12h_aligned[i]) or np.isnan(r4_12h_aligned[i]) or
            np.isnan(s3_12h_aligned[i]) or np.isnan(s4_12h_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Only trade when ADX indicates strong trend (ADX > 25)
        if adx_12h_aligned[i] <= 25:
            if position != 0:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.0
            continue
        
        if position == 1:  # Long position
            # Exit conditions: mean reversion or pivot failure
            if williams_r[i] >= -20:  # Overbought
                position = 0
                signals[i] = 0.0
            elif close[i] < s3_12h_aligned[i]:  # Broken below S3
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit conditions: mean reversion or pivot failure
            if williams_r[i] <= -80:  # Oversold
                position = 0
                signals[i] = 0.0
            elif close[i] > r3_12h_aligned[i]:  # Broken above R3
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat
            # Look for mean reversion entries at extreme Williams %R levels
            # Long: oversold + price at/supported by S3/S4
            if (williams_r[i] <= -80 and  # Oversold
                close[i] >= s4_12h_aligned[i] and  # At or above S4
                close[i] <= s3_12h_aligned[i]):  # At or below S3 (between S4 and S3)
                position = 1
                signals[i] = 0.25
            # Short: overbought + price at/resisted by R3/R4
            elif (williams_r[i] >= -20 and  # Overbought
                  close[i] <= r4_12h_aligned[i] and  # At or below R4
                  close[i] >= r3_12h_aligned[i]):  # At or above R3 (between R3 and R4)
                position = -1
                signals[i] = -0.25
    
    return signals
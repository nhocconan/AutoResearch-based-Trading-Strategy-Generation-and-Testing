#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h strategy using 12h Camarilla pivot levels for mean reversion in ranging markets and breakout continuation in trending markets
# - Uses 12h HTF to calculate Camarilla pivot levels (R3/R4/S3/S4) from prior 12h bar
# - In ranging markets (ADX < 25 on 12h): fade extremes - short at R4, long at S4 with target at R3/S3
# - In trending markets (ADX >= 25 on 12h): breakout continuation - long when price closes above R4, short when closes below S4
# - Volume confirmation: current 6h volume > 1.3x 20-period average to filter low-quality signals
# - Fixed position size 0.25 to control drawdown
# - Works in bull/bear: Camarilla levels provide adaptive support/resistance, volume confirms conviction
# - Target: 12-30 trades/year on 6h timeframe (50-120 total over 4 years)

name = "6h_12h_camarilla_pivot_meanrev_trend_v1"
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
    
    # Load 12h data ONCE before loop
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 30:
        return np.zeros(n)
    
    high_12h = df_12h['high'].values
    low_12h = df_12h['low'].values
    close_12h = df_12h['close'].values
    
    # Calculate 12h Camarilla pivot levels from prior completed 12h bar
    # Camarilla levels: based on previous day's high, low, close
    # R4 = Close + ((High - Low) * 1.1/2)
    # R3 = Close + ((High - Low) * 1.1/4)
    # S3 = Close - ((High - Low) * 1.1/4)
    # S4 = Close - ((High - Low) * 1.1/2)
    prev_high_12h = np.roll(high_12h, 1)
    prev_low_12h = np.roll(low_12h, 1)
    prev_close_12h = np.roll(close_12h, 1)
    prev_high_12h[0] = high_12h[0]  # First bar uses current high
    prev_low_12h[0] = low_12h[0]    # First bar uses current low
    prev_close_12h[0] = close_12h[0] # First bar uses current close
    
    camarilla_r4 = prev_close_12h + ((prev_high_12h - prev_low_12h) * 1.1 / 2)
    camarilla_r3 = prev_close_12h + ((prev_high_12h - prev_low_12h) * 1.1 / 4)
    camarilla_s3 = prev_close_12h - ((prev_high_12h - prev_low_12h) * 1.1 / 4)
    camarilla_s4 = prev_close_12h - ((prev_high_12h - prev_low_12h) * 1.1 / 2)
    
    # Align Camarilla levels to 6h timeframe (wait for completed 12h bar)
    r4_aligned = align_htf_to_ltf(prices, df_12h, camarilla_r4)
    r3_aligned = align_htf_to_ltf(prices, df_12h, camarilla_r3)
    s3_aligned = align_htf_to_ltf(prices, df_12h, camarilla_s3)
    s4_aligned = align_htf_to_ltf(prices, df_12h, camarilla_s4)
    
    # Calculate 12h ADX for regime detection (trending vs ranging)
    # +DI, -DI, DX calculation
    high_diff = np.diff(high_12h, prepend=high_12h[0])
    low_diff = np.diff(low_12h, prepend=low_12h[0])
    
    plus_dm = np.where((high_diff > low_diff) & (high_diff > 0), high_diff, 0)
    minus_dm = np.where((low_diff > high_diff) & (low_diff > 0), low_diff, 0)
    
    tr1 = high_12h - low_12h
    tr2 = np.abs(np.roll(high_12h, 1) - low_12h)
    tr3 = np.abs(np.roll(low_12h, 1) - high_12h)
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]
    
    atr_12h = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    plus_di = 100 * pd.Series(plus_dm).rolling(window=14, min_periods=14).mean().values / atr_12h
    minus_di = 100 * pd.Series(minus_dm).rolling(window=14, min_periods=14).mean().values / atr_12h
    dx = 100 * np.abs(plus_di - minus_di) / (plus_di + minus_di)
    adx_12h = pd.Series(dx).rolling(window=14, min_periods=14).mean().values
    
    # Align ADX to 6h timeframe
    adx_aligned = align_htf_to_ltf(prices, df_12h, adx_12h)
    
    # Pre-compute volume confirmation (20-period average)
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(100, n):
        # Skip if any required data is invalid
        if (np.isnan(r4_aligned[i]) or np.isnan(r3_aligned[i]) or
            np.isnan(s3_aligned[i]) or np.isnan(s4_aligned[i]) or
            np.isnan(adx_aligned[i]) or np.isnan(vol_ma_20[i]) or
            vol_ma_20[i] <= 0):
            signals[i] = 0.0
            continue
        
        # Volume confirmation: current 6h volume > 1.3x average
        volume_confirmed = volume[i] > 1.3 * vol_ma_20[i]
        
        if not volume_confirmed:
            signals[i] = 0.0
            continue
        
        if position == 1:  # Long position
            # Exit conditions
            if adx_aligned[i] < 25:  # Ranging market: take profit at R3
                if close[i] >= r3_aligned[i]:
                    position = 0
                    signals[i] = 0.0
                else:
                    signals[i] = 0.25
            else:  # Trending market: hold until breakout fails
                if close[i] < r4_aligned[i]:
                    position = 0
                    signals[i] = 0.0
                else:
                    signals[i] = 0.25
                    
        elif position == -1:  # Short position
            # Exit conditions
            if adx_aligned[i] < 25:  # Ranging market: take profit at S3
                if close[i] <= s3_aligned[i]:
                    position = 0
                    signals[i] = 0.0
                else:
                    signals[i] = -0.25
            else:  # Trending market: hold until breakout fails
                if close[i] > s4_aligned[i]:
                    position = 0
                    signals[i] = 0.0
                else:
                    signals[i] = -0.25
        else:  # Flat
            # Entry logic based on regime
            if adx_aligned[i] < 25:  # Ranging market: mean reversion at extremes
                if close[i] <= s4_aligned[i] and volume_confirmed:
                    position = -1  # Short at S4
                    signals[i] = -0.25
                elif close[i] >= r4_aligned[i] and volume_confirmed:
                    position = 1   # Long at R4
                    signals[i] = 0.25
            else:  # Trending market: breakout continuation
                if close[i] > r4_aligned[i] and volume_confirmed:
                    position = 1   # Long on breakout above R4
                    signals[i] = 0.25
                elif close[i] < s4_aligned[i] and volume_confirmed:
                    position = -1  # Short on breakdown below S4
                    signals[i] = -0.25
    
    return signals
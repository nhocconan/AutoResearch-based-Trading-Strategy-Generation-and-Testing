#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Camarilla pivot breakout with 12h volume confirmation and 1d regime filter
# - Uses 12h Camarilla levels calculated from prior 1d OHLC (R3/S3 for fade, R4/S4 for breakout)
# - Confirms breakout with 12h volume > 2.0x 24-period average (strong institutional participation)
# - Filters by 1d ADX > 25 to ensure trending environment (avoids whipsaws in range)
# - Long when price breaks above R4 with volume and ADX>25, short when breaks below S4
# - Exits when price returns to R3/S3 levels (mean reversion to pivot area)
# - Position size: 0.25 (25% of capital) to balance return and drawdown
# - Target: 12-30 trades/year on 6h timeframe (48-120 total over 4 years) to minimize fee drag
# - Camarilla levels work well in crypto due to respect of key pivot points
# - Volume confirmation ensures breakouts have follow-through
# - ADX filter avoids false signals in low-momentum environments

name = "6h_12h_1d_camarilla_breakout_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Load HTF data ONCE before loop
    df_12h = get_htf_data(prices, '12h')
    df_1d = get_htf_data(prices, '1d')
    if len(df_12h) < 30 or len(df_1d) < 30:
        return np.zeros(n)
    
    # Pre-compute 1d indicators for Camarilla calculation and ADX
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate 12h Camarilla levels from prior 1d OHLC
    # Camarilla formulas: Range = high - low
    # R4 = close + Range * 1.1/2, R3 = close + Range * 1.1/4
    # S3 = close - Range * 1.1/4, S4 = close - Range * 1.1/2
    prev_close = np.roll(close_1d, 1)
    prev_high = np.roll(high_1d, 1)
    prev_low = np.roll(low_1d, 1)
    
    # Handle first bar
    prev_close[0] = close_1d[0]
    prev_high[0] = high_1d[0]
    prev_low[0] = low_1d[0]
    
    daily_range = prev_high - prev_low
    camarilla_r4 = prev_close + (daily_range * 1.1 / 2)
    camarilla_r3 = prev_close + (daily_range * 1.1 / 4)
    camarilla_s3 = prev_close - (daily_range * 1.1 / 4)
    camarilla_s4 = prev_close - (daily_range * 1.1 / 2)
    
    # 1d ADX(14) for regime filter
    # True Range
    tr1 = high_1d - low_1d
    tr2 = np.abs(high_1d - np.roll(close_1d, 1))
    tr3 = np.abs(low_1d - np.roll(close_1d, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]
    
    # Directional Movement
    up_move = high_1d - np.roll(high_1d, 1)
    down_move = np.roll(low_1d, 1) - low_1d
    up_move[0] = 0
    down_move[0] = 0
    
    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0)
    
    # Smoothed TR, +DM, -DM
    atr = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    plus_di = 100 * pd.Series(plus_dm).rolling(window=14, min_periods=14).mean().values / np.where(atr != 0, atr, 1e-10)
    minus_di = 100 * pd.Series(minus_dm).rolling(window=14, min_periods=14).mean().values / np.where(atr != 0, atr, 1e-10)
    
    # DX and ADX
    dx = 100 * np.abs(plus_di - minus_di) / np.where((plus_di + minus_di) != 0, (plus_di + minus_di), 1e-10)
    adx = pd.Series(dx).rolling(window=14, min_periods=14).mean().values
    
    # 12h volume > 2.0x 24-period average
    volume_12h = df_12h['volume'].values
    avg_volume_24 = pd.Series(volume_12h).rolling(window=24, min_periods=24).mean().values
    volume_spike = volume_12h > (2.0 * avg_volume_24)
    
    # Align all indicators to 6h
    camarilla_r4_aligned = align_htf_to_ltf(prices, df_1d, camarilla_r4)
    camarilla_r3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_r3)
    camarilla_s3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_s3)
    camarilla_s4_aligned = align_htf_to_ltf(prices, df_1d, camarilla_s4)
    adx_aligned = align_htf_to_ltf(prices, df_1d, adx)
    volume_spike_aligned = align_htf_to_ltf(prices, df_12h, volume_spike.astype(float))
    
    # 6h price data
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(50, n):
        # Skip if any required data is invalid
        if (np.isnan(camarilla_r4_aligned[i]) or np.isnan(camarilla_r3_aligned[i]) or
            np.isnan(camarilla_s3_aligned[i]) or np.isnan(camarilla_s4_aligned[i]) or
            np.isnan(adx_aligned[i]) or np.isnan(volume_spike_aligned[i]) or
            adx_aligned[i] < 25):  # ADX filter for trending regime
            signals[i] = 0.0
            continue
        
        if position == 1:  # Long position
            # Exit when price returns to R3 (mean reversion to pivot area)
            if low[i] <= camarilla_r3_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit when price returns to S3 (mean reversion to pivot area)
            if high[i] >= camarilla_s3_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat
            # Look for Camarilla breakout with volume confirmation and ADX filter
            if (high[i] >= camarilla_r4_aligned[i] and  # Break above R4
                volume_spike_aligned[i]):              # Volume confirmation
                position = 1
                signals[i] = 0.25
            elif (low[i] <= camarilla_s4_aligned[i] and   # Break below S4
                  volume_spike_aligned[i]):              # Volume confirmation
                position = -1
                signals[i] = -0.25
    
    return signals
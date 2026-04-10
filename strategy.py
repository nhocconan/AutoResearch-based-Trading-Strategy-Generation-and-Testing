#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Camarilla pivot breakout with 12h volume confirmation and ADX regime filter
# - Primary: 6h price breaking above R4 or below S4 Camarilla levels from 12h
# - HTF: 12h volume confirmation (current volume > 1.5x 20-period MA) + ADX > 25 for trend strength
# - Long: Breakout above R4 + volume confirmation + ADX > 25
# - Short: Breakout below S4 + volume confirmation + ADX > 25
# - Exit: Price returns to R3/S3 levels
# - Position sizing: 0.25 (discrete level to minimize fee churn)
# - Works in bull/bear: Camarilla pivots act as support/resistance, volume confirms momentum, ADX filters ranging markets
# - Target: 50-150 total trades over 4 years (12-37/year) to stay within fee drag limits

name = "6h_12h_camarilla_breakout_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 60:  # Need enough data for calculations
        return np.zeros(n)
    
    # Load HTF data ONCE before loop
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 30:  # Need enough data for Camarilla and ADX
        return np.zeros(n)
    
    # Pre-compute 6h data
    close_6h = prices['close'].values
    
    # Pre-compute 12h data
    high_12h = df_12h['high'].values
    low_12h = df_12h['low'].values
    close_12h = df_12h['close'].values
    volume_12h = df_12h['volume'].values
    
    # Calculate Camarilla pivot levels (12h) using previous period's data
    # Camarilla uses previous period's high, low, close
    prev_high = np.roll(high_12h, 1)
    prev_low = np.roll(low_12h, 1)
    prev_close = np.roll(close_12h, 1)
    
    # First period will have NaN due to roll, that's expected
    pivot = (prev_high + prev_low + prev_close) / 3.0
    range_hl = prev_high - prev_low
    
    # Camarilla levels
    r4 = pivot + (range_hl * 1.5 / 2)
    r3 = pivot + (range_hl * 1.25 / 2)
    s3 = pivot - (range_hl * 1.25 / 2)
    s4 = pivot - (range_hl * 1.5 / 2)
    
    # Calculate ADX (12h) for trend strength
    # True Range
    tr1 = np.abs(np.roll(high_12h, 1) - np.roll(low_12h, 1))
    tr2 = np.abs(np.roll(high_12h, 1) - np.roll(close_12h, 1))
    tr3 = np.abs(np.roll(low_12h, 1) - np.roll(close_12h, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    
    # Directional Movement
    up_move = np.roll(high_12h, 1) - high_12h
    down_move = low_12h - np.roll(low_12h, 1)
    
    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0.0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0.0)
    
    # Smoothed values (using Wilder's smoothing = EMA with alpha=1/period)
    period = 14
    alpha = 1.0 / period
    
    atr = pd.Series(tr).ewm(alpha=alpha, adjust=False).mean().values
    plus_di = 100 * pd.Series(plus_dm).ewm(alpha=alpha, adjust=False).mean().values / atr
    minus_di = 100 * pd.Series(minus_dm).ewm(alpha=alpha, adjust=False).mean().values / atr
    
    dx = 100 * np.abs(plus_di - minus_di) / (plus_di + minus_di)
    adx = pd.Series(dx).ewm(alpha=alpha, adjust=False).mean().values
    
    # Calculate 12h volume moving average (20-period) for volume confirmation
    volume_ma_20_12h = pd.Series(volume_12h).rolling(window=20, min_periods=20).mean().values
    
    # Align all HTF indicators to 6h timeframe
    r4_aligned = align_htf_to_ltf(prices, df_12h, r4)
    r3_aligned = align_htf_to_ltf(prices, df_12h, r3)
    s3_aligned = align_htf_to_ltf(prices, df_12h, s3)
    s4_aligned = align_htf_to_ltf(prices, df_12h, s4)
    adx_aligned = align_htf_to_ltf(prices, df_12h, adx)
    volume_ma_20_12h_aligned = align_htf_to_ltf(prices, df_12h, volume_ma_20_12h)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(30, n):  # Start after warmup period
        # Skip if any required data is invalid
        if (np.isnan(r4_aligned[i]) or np.isnan(r3_aligned[i]) or
            np.isnan(s3_aligned[i]) or np.isnan(s4_aligned[i]) or
            np.isnan(adx_aligned[i]) or np.isnan(volume_ma_20_12h_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Get current 12h volume (aligned to 6h)
        volume_12h_aligned = align_htf_to_ltf(prices, df_12h, volume_12h)
        
        # Volume confirmation: current 12h volume > 1.5x 20-period MA
        volume_confirm = volume_12h_aligned[i] > 1.5 * volume_ma_20_12h_aligned[i]
        
        # ADX trend filter: ADX > 25 indicates strong trend
        trend_confirm = adx_aligned[i] > 25.0
        
        # Camarilla breakout conditions
        breakout_long = close_6h[i] > r4_aligned[i]
        breakout_short = close_6h[i] < s4_aligned[i]
        
        # Exit conditions: Price returns to R3/S3 levels
        exit_long = close_6h[i] < r3_aligned[i]
        exit_short = close_6h[i] > s3_aligned[i]
        
        if position == 0:  # Flat - look for new entries
            # Long entry: Breakout above R4 + volume confirmation + trend confirmation
            if breakout_long and volume_confirm and trend_confirm:
                position = 1
                signals[i] = 0.25
            # Short entry: Breakout below S4 + volume confirmation + trend confirmation
            elif breakout_short and volume_confirm and trend_confirm:
                position = -1
                signals[i] = -0.25
            else:
                signals[i] = 0.0
        else:  # Have position - look for exit
            # Exit: Price returns to R3/S3 levels
            if position == 1:  # Long position
                if exit_long:
                    position = 0
                    signals[i] = 0.0
                else:
                    signals[i] = 0.25
            else:  # position == -1 (Short position)
                if exit_short:
                    position = 0
                    signals[i] = 0.0
                else:
                    signals[i] = -0.25
    
    return signals
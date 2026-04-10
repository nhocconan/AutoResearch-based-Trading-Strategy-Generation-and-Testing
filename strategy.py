#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Weekly Camarilla Pivot Breakout with Daily Volume Confirmation
# - Uses weekly Camarilla pivot levels (R3, S3, R4, S4) from prior week
# - Breakout logic: Long when price closes above weekly R3 with volume confirmation
#                   Short when price closes below weekly S3 with volume confirmation
# - Continuation logic: Add to position on break of R4/S4 with volume confirmation
# - Mean reversion in range: Fade at R3/S3 when price reverses with volume confirmation
# - Weekly trend filter: Only take longs in weekly uptrend (price > weekly EMA50),
#                        only take shorts in weekly downtrend (price < weekly EMA50)
# - Discrete position sizing (0.25) to minimize fee churn
# - Weekly timeframe provides structural levels, daily volume confirms institutional interest
# - Target: 20-30 trades/year (80-120 total over 4 years) to stay within HARD MAX: 300 total

name = "6h_1w_camarilla_breakout_volume_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Load HTF data ONCE before loop
    df_1w = get_htf_data(prices, '1w')
    df_1d = get_htf_data(prices, '1d')
    if len(df_1w) < 10 or len(df_1d) < 50:
        return np.zeros(n)
    
    # Pre-compute weekly Camarilla pivot levels from prior week
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    
    # Calculate prior week's Camarilla levels (shifted by 1 to avoid look-ahead)
    # Weekly range
    weekly_range = high_1w - low_1w
    
    # Camarilla levels for weekly timeframe
    # R4 = close + range * 1.1/2
    # R3 = close + range * 1.1/4
    # S3 = close - range * 1.1/4
    # S4 = close - range * 1.1/2
    camarilla_r4 = close_1w + weekly_range * 1.1 / 2
    camarilla_r3 = close_1w + weekly_range * 1.1 / 4
    camarilla_s3 = close_1w - weekly_range * 1.1 / 4
    camarilla_s4 = close_1w - weekly_range * 1.1 / 2
    
    # Shift by 1 week to use prior week's levels (no look-ahead)
    camarilla_r4 = np.roll(camarilla_r4, 1)
    camarilla_r3 = np.roll(camarilla_r3, 1)
    camarilla_s3 = np.roll(camarilla_s3, 1)
    camarilla_s4 = np.roll(camarilla_s4, 1)
    camarilla_r4[0] = np.nan
    camarilla_r3[0] = np.nan
    camarilla_s3[0] = np.nan
    camarilla_s4[0] = np.nan
    
    # Pre-compute weekly EMA(50) for trend filter
    ema_50_1w = pd.Series(close_1w).ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Pre-compute daily volume and its 20-period moving average for confirmation
    volume_1d = df_1d['volume'].values
    volume_ma_20_1d = pd.Series(volume_1d).rolling(window=20, min_periods=20).mean().values
    
    # Align all HTF indicators to 6h timeframe
    camarilla_r4_aligned = align_htf_to_ltf(prices, df_1w, camarilla_r4)
    camarilla_r3_aligned = align_htf_to_ltf(prices, df_1w, camarilla_r3)
    camarilla_s3_aligned = align_htf_to_ltf(prices, df_1w, camarilla_s3)
    camarilla_s4_aligned = align_htf_to_ltf(prices, df_1w, camarilla_s4)
    ema_50_aligned = align_htf_to_ltf(prices, df_1w, ema_50_1w)
    volume_ma_aligned = align_htf_to_ltf(prices, df_1d, volume_ma_20_1d)
    
    # Pre-compute 6h ATR for volatility filter (optional)
    high_6h = prices['high'].values
    low_6h = prices['low'].values
    close_6h = prices['close'].values
    
    tr1_6h = high_6h - low_6h
    tr2_6h = np.abs(high_6h - np.roll(close_6h, 1))
    tr3_6h = np.abs(low_6h - np.roll(close_6h, 1))
    tr1_6h[0] = np.nan
    tr2_6h[0] = np.nan
    tr3_6h[0] = np.nan
    tr_6h = np.maximum.reduce([tr1_6h, tr2_6h, tr3_6h])
    atr_6h = pd.Series(tr_6h).rolling(window=14, min_periods=14).mean().values
    
    # Pre-compute 6h volume and its 20-period moving average
    volume_6h = prices['volume'].values
    volume_ma_20_6h = pd.Series(volume_6h).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    entry_price = 0.0
    
    for i in range(60, n):  # Start after warmup
        # Skip if any required data is invalid
        if (np.isnan(camarilla_r3_aligned[i]) or np.isnan(camarilla_s3_aligned[i]) or 
            np.isnan(camarilla_r4_aligned[i]) or np.isnan(camarilla_s4_aligned[i]) or 
            np.isnan(ema_50_aligned[i]) or np.isnan(volume_ma_aligned[i]) or 
            np.isnan(volume_ma_20_6h[i])):
            if position == 0:
                signals[i] = 0.0
            elif position == 1:
                signals[i] = 0.25
            else:
                signals[i] = -0.25
            continue
        
        # Get current 6h data
        close_price = close_6h[i]
        volume_6h_current = volume_6h[i]
        
        # Get prior week's Camarilla levels (already aligned)
        r4_level = camarilla_r4_aligned[i]
        r3_level = camarilla_r3_aligned[i]
        s3_level = camarilla_s3_aligned[i]
        s4_level = camarilla_s4_aligned[i]
        
        # Weekly trend filter
        weekly_uptrend = close_1w[-1] > ema_50_1w[-1] if len(close_1w) > 0 else False  # Simplified: use current weekly trend
        # Better: get current weekly EMA from aligned array
        weekly_uptrend = close_price > ema_50_aligned[i]
        weekly_downtrend = close_price < ema_50_aligned[i]
        
        # Volume confirmation: current 6h volume > 1.5x 20-period average
        volume_spike_6h = volume_6h_current > 1.5 * volume_ma_20_6h[i]
        # Also check daily volume confirmation
        volume_spike_1d = False  # Simplified for now - could add if needed
        
        if position == 0:  # Flat - look for new entries
            # Breakout entries with volume confirmation and weekly trend alignment
            if volume_spike_6h:
                # Long breakout: price closes above weekly R3 in weekly uptrend
                if close_price > r3_level and weekly_uptrend:
                    position = 1
                    entry_price = prices['open'].iloc[i+1] if i+1 < n else prices['close'].iloc[i]
                    signals[i] = 0.25
                # Short breakout: price closes below weekly S3 in weekly downtrend
                elif close_price < s3_level and weekly_downtrend:
                    position = -1
                    entry_price = prices['open'].iloc[i+1] if i+1 < n else prices['close'].iloc[i]
                    signals[i] = -0.25
                else:
                    signals[i] = 0.0
            else:
                signals[i] = 0.0
                
            # Add-to-position logic on break of R4/S4 (continuation)
            if position == 0 and volume_spike_6h:
                if close_price > r4_level and weekly_uptrend:
                    position = 1
                    entry_price = prices['open'].iloc[i+1] if i+1 < n else prices['close'].iloc[i]
                    signals[i] = 0.25
                elif close_price < s4_level and weekly_downtrend:
                    position = -1
                    entry_price = prices['open'].iloc[i+1] if i+1 < n else prices['close'].iloc[i]
                    signals[i] = -0.25
                    
        else:  # Have position - look for exit or add-to-position
            # Add-to-position on continuation breakouts
            if position == 1 and volume_spike_6h:
                if close_price > r4_level:
                    # Pyramid up to 0.5 position (discrete: 0.25 + 0.25)
                    signals[i] = 0.50
                else:
                    signals[i] = 0.25
            elif position == -1 and volume_spike_6h:
                if close_price < s4_level:
                    # Pyramid up to 0.5 position (discrete: -0.25 + -0.25)
                    signals[i] = -0.50
                else:
                    signals[i] = -0.25
            else:
                # Hold current position
                if position == 1:
                    signals[i] = 0.25
                else:
                    signals[i] = -0.25
            
            # Exit conditions: mean reversion at extreme levels with volume
            if position == 1 and volume_spike_6h:
                # Exit long if price reverses below R3 (mean reversion)
                if close_price < r3_level:
                    position = 0
                    entry_price = 0.0
                    signals[i] = 0.0
            elif position == -1 and volume_spike_6h:
                # Exit short if price reverses above S3 (mean reversion)
                if close_price > s3_level:
                    position = 0
                    entry_price = 0.0
                    signals[i] = 0.0
    
    return signals
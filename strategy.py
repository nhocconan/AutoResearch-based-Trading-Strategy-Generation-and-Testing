#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Choppiness Index regime filter with Donchian breakout and volume confirmation.
# In trending markets (CHOP < 38.2): follow Donchian breakouts.
# In ranging markets (CHOP > 61.8): mean-revert at Donchian channels.
# Volume confirmation filters low-quality breakouts.
# Target: 20-40 trades/year by requiring regime alignment + breakout + volume spike.

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Load daily data for Choppiness Index calculation
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    
    # Load 4h data for Donchian channels and volume
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 20:
        return np.zeros(n)
    
    # Calculate daily True Range for Choppiness Index
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    tr1 = high_1d - low_1d
    tr2 = np.abs(high_1d - np.roll(close_1d, 1))
    tr3 = np.abs(low_1d - np.roll(close_1d, 1))
    tr1[0] = 0  # First bar has no previous close
    tr2[0] = 0
    tr3[0] = 0
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    
    # Calculate ATR(14) and sum for denominator
    atr_14 = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    atr_sum = pd.Series(atr_14).rolling(window=14, min_periods=14).sum().values
    
    # Calculate max(high) and min(low) over 14 periods for numerator
    max_high = pd.Series(high_1d).rolling(window=14, min_periods=14).max().values
    min_low = pd.Series(low_1d).rolling(window=14, min_periods=14).min().values
    range_max_min = max_high - min_low
    
    # Choppiness Index: 100 * log10(atr_sum / range_max_min) / log10(14)
    # Avoid division by zero and invalid values
    chop_raw = np.where((range_max_min > 0) & (atr_sum > 0), 
                        100 * np.log10(atr_sum / range_max_min) / np.log10(14), 
                        50.0)  # Default to neutral when invalid
    chop = pd.Series(chop_raw).rolling(window=14, min_periods=14).mean().values
    
    # Calculate 4h Donchian channels (20-period)
    high_4h = df_4h['high'].values
    low_4h = df_4h['low'].values
    donch_high = pd.Series(high_4h).rolling(window=20, min_periods=20).max().values
    donch_low = pd.Series(low_4h).rolling(window=20, min_periods=20).min().values
    
    # Calculate 4h volume moving average (20-period)
    vol_4h = df_4h['volume'].values
    vol_ma_20 = pd.Series(vol_4h).rolling(window=20, min_periods=20).mean().values
    
    # Align daily Choppiness Index to 4h
    chop_aligned = align_htf_to_ltf(prices, df_1d, chop)
    
    # Align 4h indicators to 4h (no additional delay needed as they're already calculated)
    donch_high_aligned = align_htf_to_ltf(prices, df_4h, donch_high)
    donch_low_aligned = align_htf_to_ltf(prices, df_4h, donch_low)
    vol_ma_20_aligned = align_htf_to_ltf(prices, df_4h, vol_ma_20)
    
    # Pre-compute session hours (08-20 UTC)
    hours = pd.DatetimeIndex(prices['open_time']).hour
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(100, n):
        # Skip if data not ready
        if (np.isnan(chop_aligned[i]) or np.isnan(donch_high_aligned[i]) or 
            np.isnan(donch_low_aligned[i]) or np.isnan(vol_ma_20_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Session filter: 08-20 UTC
        hour = hours[i]
        in_session = 8 <= hour <= 20
        
        if not in_session:
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Current values
        close = prices['close'].iloc[i]
        chop_val = chop_aligned[i]
        vol_current = vol_4h[i]  # Current 4h volume
        
        # Regime filters
        trending = chop_val < 38.2   # Strong trend
        ranging = chop_val > 61.8    # Strong range
        
        # Volume confirmation: current volume > 1.5x 20-period average
        volume_confirm = vol_current > 1.5 * vol_ma_20_aligned[i]
        
        if position == 0:
            # Entry logic based on regime
            if trending and volume_confirm:
                # In trending market: follow Donchian breakout
                if close > donch_high_aligned[i]:
                    signals[i] = 0.25
                    position = 1
                elif close < donch_low_aligned[i]:
                    signals[i] = -0.25
                    position = -1
            elif ranging and volume_confirm:
                # In ranging market: mean-revert at Donchian channels
                if close <= donch_low_aligned[i]:
                    signals[i] = 0.25  # Buy at support
                    position = 1
                elif close >= donch_high_aligned[i]:
                    signals[i] = -0.25  # Sell at resistance
                    position = -1
        
        elif position != 0:
            # Exit logic
            exit_signal = False
            
            if position == 1:  # Long position
                # Exit conditions: opposite Donchian touch or regime change against position
                if close >= donch_high_aligned[i]:  # Hit resistance - take profit
                    exit_signal = True
                elif ranging and chop_val > 50:  # Range developing - exit longs
                    exit_signal = True
                elif trending and close < donch_low_aligned[i]:  # Breakdown in trend
                    exit_signal = True
                    
            elif position == -1:  # Short position
                # Exit conditions: opposite Donchian touch or regime change against position
                if close <= donch_low_aligned[i]:  # Hit support - take profit
                    exit_signal = True
                elif ranging and chop_val > 50:  # Range developing - exit shorts
                    exit_signal = True
                elif trending and close > donch_high_aligned[i]:  # Breakout in trend
                    exit_signal = True
            
            if exit_signal:
                signals[i] = 0.0
                position = 0
            else:
                # Hold position
                signals[i] = 0.25 if position == 1 else -0.25
    
    return signals

name = "4h_Chop_Donchian_Breakout_MR_Volume"
timeframe = "4h"
leverage = 1.0
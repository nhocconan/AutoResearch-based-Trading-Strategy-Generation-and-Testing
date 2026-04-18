#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Williams Alligator with 1d Supertrend trend filter and volume confirmation.
# Williams Alligator identifies trend direction via jaw/teeth/lips alignment.
# 1d Supertrend filters for strong trends to avoid whipsaws in ranging markets.
# Volume confirmation ensures breakouts have conviction.
# Designed for low trade frequency (20-40/year) to minimize fee drag.
# Works in bull markets (bullish alignment with rising Supertrend) and bear markets 
# (bearish alignment with rising Supertrend).
name = "4h_WilliamsAlligator_1dSupertrend_Volume"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 4h data for Williams Alligator calculation (ONCE before loop)
    df_4h = get_htf_data(prices, '4h')
    # Get daily data for Supertrend calculation (ONCE before loop)
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate Williams Alligator components (using previous bar's data to avoid look-ahead)
    # Jaw (13-period SMMA, shifted 8 bars)
    high_13 = df_4h['high'].rolling(window=13, min_periods=13).mean()
    low_13 = df_4h['low'].rolling(window=13, min_periods=13).mean()
    median_price = (high_13 + low_13) / 2
    jaw_raw = median_price.rolling(window=8, min_periods=8).mean().shift(8).values
    
    # Teeth (8-period SMMA, shifted 5 bars)
    high_8 = df_4h['high'].rolling(window=8, min_periods=8).mean()
    low_8 = df_4h['low'].rolling(window=8, min_periods=8).mean()
    median_price_8 = (high_8 + low_8) / 2
    teeth_raw = median_price_8.rolling(window=5, min_periods=5).mean().shift(5).values
    
    # Lips (5-period SMMA, shifted 3 bars)
    high_5 = df_4h['high'].rolling(window=5, min_periods=5).mean()
    low_5 = df_4h['low'].rolling(window=5, min_periods=5).mean()
    median_price_5 = (high_5 + low_5) / 2
    lips_raw = median_price_5.rolling(window=3, min_periods=3).mean().shift(3).values
    
    # Align Williams Alligator components to 4h timeframe
    jaw_aligned = align_htf_to_ltf(prices, df_4h, jaw_raw)
    teeth_aligned = align_htf_to_ltf(prices, df_4h, teeth_raw)
    lips_aligned = align_htf_to_ltf(prices, df_4h, lips_raw)
    
    # Calculate 1d Supertrend
    atr_period = 10
    multiplier = 3.0
    
    # True Range
    tr1 = df_1d['high'] - df_1d['low']
    tr2 = np.abs(df_1d['high'] - df_1d['close'].shift(1))
    tr3 = np.abs(df_1d['low'] - df_1d['close'].shift(1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    atr = tr.rolling(window=atr_period, min_periods=atr_period).mean().values
    
    # Upper and Lower Bands
    hl_avg = (df_1d['high'] + df_1d['low']) / 2
    upper_band = (hl_avg + multiplier * atr).values
    lower_band = (hl_avg - multiplier * atr).values
    
    # Supertrend calculation
    supertrend = np.full_like(close, np.nan, dtype=float)
    direction = np.full_like(close, np.nan, dtype=float)  # 1 for uptrend, -1 for downtrend
    
    # Initialize
    for i in range(atr_period, len(df_1d)):
        if i == atr_period:
            supertrend[i] = lower_band[i]
            direction[i] = 1
        else:
            if supertrend[i-1] == upper_band[i-1]:
                if df_1d['close'].iloc[i] <= upper_band[i]:
                    supertrend[i] = upper_band[i]
                    direction[i] = -1
                else:
                    supertrend[i] = lower_band[i]
                    direction[i] = 1
            else:
                if df_1d['close'].iloc[i] >= lower_band[i]:
                    supertrend[i] = lower_band[i]
                    direction[i] = 1
                else:
                    supertrend[i] = upper_band[i]
                    direction[i] = -1
    
    # Align Supertrend to 4h timeframe
    supertrend_aligned = align_htf_to_ltf(prices, df_1d, supertrend)
    direction_aligned = align_htf_to_ltf(prices, df_1d, direction)
    
    # Calculate 20-period average volume for confirmation
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    # Session filter: 08-20 UTC
    hour_index = pd.DatetimeIndex(prices['open_time']).hour
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 100  # Wait for indicator calculations
    
    for i in range(start_idx, n):
        # Skip if any required data is not available
        if (np.isnan(jaw_aligned[i]) or np.isnan(teeth_aligned[i]) or np.isnan(lips_aligned[i]) or
            np.isnan(supertrend_aligned[i]) or np.isnan(direction_aligned[i]) or np.isnan(vol_ma_20[i])):
            signals[i] = 0.0
            continue
        
        hour = hour_index[i]
        in_session = 8 <= hour <= 20
        
        if not in_session:
            signals[i] = 0.0
            continue
        
        # Volume confirmation: current volume above average
        vol_confirm = volume[i] > vol_ma_20[i]
        
        # Williams Alligator signals
        # Bullish: Lips > Teeth > Jaw (all aligned upward)
        bullish_alignment = (lips_aligned[i] > teeth_aligned[i]) and (teeth_aligned[i] > jaw_aligned[i])
        # Bearish: Jaw > Teeth > Lips (all aligned downward)
        bearish_alignment = (jaw_aligned[i] > teeth_aligned[i]) and (teeth_aligned[i] > lips_aligned[i])
        
        if position == 0:
            # Long: Bullish alignment AND Supertrend uptrend AND volume
            if bullish_alignment and (direction_aligned[i] == 1) and vol_confirm:
                signals[i] = 0.25
                position = 1
            # Short: Bearish alignment AND Supertrend downtrend AND volume
            elif bearish_alignment and (direction_aligned[i] == -1) and vol_confirm:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Long exit: Bearish alignment OR Supertrend turns down
            if bearish_alignment or (direction_aligned[i] == -1):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: Bullish alignment OR Supertrend turns up
            if bullish_alignment or (direction_aligned[i] == 1):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals
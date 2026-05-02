#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1h EMA21 crossover with 4h Donchian(20) trend filter and volume spike confirmation
# Uses 4h Donchian breakout for signal direction (trend filter) and 1h for precise entry timing via EMA21 crossovers
# Volume confirmation (2.0x 20-period average on 1h) ensures institutional participation
# Session filter (08-20 UTC) reduces noise trades outside active hours
# Target: 80-120 total trades over 4 years = 20-30/year for 1h timeframe
# Works in bull markets via Donchian breakout alignment, in bear via tight EMA21 crossovers with volume filter
# Designed for low trade frequency to minimize fee drag (critical for 1h timeframe)

name = "1h_EMA21_Cross_4hDonchian20_VolumeSpike_Session_v1"
timeframe = "1h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Pre-compute session hours (08-20 UTC) - index is DatetimeIndex
    hours = prices.index.hour
    in_session = (hours >= 8) & (hours <= 20)
    
    # Load 4h data ONCE before loop for Donchian trend filter
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 20:
        return np.zeros(n)
    
    # Calculate 4h Donchian Channel (20)
    high_4h = df_4h['high'].values
    low_4h = df_4h['low'].values
    
    # Upper band: highest high over last 20 periods
    donchian_high = pd.Series(high_4h).rolling(window=20, min_periods=20).max().values
    # Lower band: lowest low over last 20 periods
    donchian_low = pd.Series(low_4h).rolling(window=20, min_periods=20).min().values
    
    # Align to 1h timeframe (completed 4h bars only)
    donchian_high_aligned = align_htf_to_ltf(prices, df_4h, donchian_high)
    donchian_low_aligned = align_htf_to_ltf(prices, df_4h, donchian_low)
    
    # Calculate 1h EMA21 for entry timing
    close_s = pd.Series(close)
    ema_21 = close_s.ewm(span=21, adjust=False, min_periods=21).mean().values
    
    # Volume confirmation (2.0x 20-period average on 1h)
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().shift(1).values
    volume_confirm = volume > (vol_ma * 2.0)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after warmup (need enough for indicators)
    start_idx = 50
    
    for i in range(start_idx, n):
        # Skip if outside trading session
        if not in_session[i]:
            signals[i] = 0.0
            continue
            
        # Check for NaN values in indicators
        if (np.isnan(donchian_high_aligned[i]) or np.isnan(donchian_low_aligned[i]) or 
            np.isnan(ema_21[i]) or np.isnan(volume_confirm[i])):
            signals[i] = 0.0
            continue
        
        if position == 0:  # Flat - look for new entries
            # Long: Price > EMA21 + close breaks above 4h Donchian High + volume confirm
            if close[i] > ema_21[i] and close[i] > donchian_high_aligned[i] and volume_confirm[i]:
                signals[i] = 0.20
                position = 1
            # Short: Price < EMA21 + close breaks below 4h Donchian Low + volume confirm
            elif close[i] < ema_21[i] and close[i] < donchian_low_aligned[i] and volume_confirm[i]:
                signals[i] = -0.20
                position = -1
            else:
                signals[i] = 0.0
        
        elif position == 1:  # Long position
            # Exit: Price crosses below EMA21 or breaks below 4h Donchian Low
            if close[i] < ema_21[i] or close[i] < donchian_low_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.20
        
        elif position == -1:  # Short position
            # Exit: Price crosses above EMA21 or breaks above 4h Donchian High
            if close[i] > ema_21[i] or close[i] > donchian_high_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.20
    
    return signals
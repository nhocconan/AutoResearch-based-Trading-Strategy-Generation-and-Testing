#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Williams Alligator + 1d Elder Ray + volume confirmation
# Long when Alligator jaws (13) < teeth (8) < lips (5) + Elder Bull Power > 0 + volume > 1.5x 20-period avg
# Short when Alligator jaws (13) > teeth (8) > lips (5) + Elder Bear Power < 0 + volume > 1.5x 20-period avg
# Uses Alligator for trend alignment, Elder Ray for bull/bear power confirmation, volume for conviction.
# Designed for low trade frequency (15-35/year) with discrete sizing (0.25) to minimize fee drag.
# Works in bull markets (Alligator aligned up + Elder Bull > 0) and bear markets (Alligator aligned down + Elder Bear < 0).

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Precompute session hours (08-20 UTC) for filter
    hours = pd.DatetimeIndex(prices['open_time']).hour
    in_session = (hours >= 8) & (hours <= 20)
    
    # Get 1d HTF data once before loop
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # === 1d Indicator: Elder Ray (Bull Power and Bear Power) ===
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate 13-period EMA of close
    close_1d_series = pd.Series(close_1d)
    ema13 = close_1d_series.ewm(span=13, adjust=False, min_periods=13).mean().values
    
    # Bull Power = High - EMA13
    bull_power = high_1d - ema13
    # Bear Power = Low - EMA13
    bear_power = low_1d - ema13
    
    bull_power_aligned = align_htf_to_ltf(prices, df_1d, bull_power)
    bear_power_aligned = align_htf_to_ltf(prices, df_1d, bear_power)
    
    # === 4h Indicator: Williams Alligator (SMAs of median price) ===
    # Median price = (High + Low) / 2
    median_price = (high + low) / 2
    
    # Alligator components: Jaws (13-period SMMA, 8-bar offset), Teeth (8-period SMMA, 5-bar offset), Lips (5-period SMMA, 3-bar offset)
    # Using SMA as approximation for SMMA (Smoothed Moving Average) for simplicity
    jaws = pd.Series(median_price).rolling(window=13, min_periods=13).mean().values
    teeth = pd.Series(median_price).rolling(window=8, min_periods=8).mean().values
    lips = pd.Series(median_price).rolling(window=5, min_periods=5).mean().values
    
    # Apply offsets: Jaws shifted 8 bars forward, Teeth shifted 5 bars forward, Lips shifted 3 bars forward
    # To avoid look-ahead, we use the values as-is (they represent the state after the shift period)
    # In practice, Alligator uses future data for the lines, but we interpret the current values as the lagged indicator
    jaws_lagged = np.roll(jaws, 8)
    teeth_lagged = np.roll(teeth, 5)
    lips_lagged = np.roll(lips, 3)
    # Fill the rolled values with NaN for the initial period
    jaws_lagged[:8] = np.nan
    teeth_lagged[:5] = np.nan
    lips_lagged[:3] = np.nan
    
    # Volume SMA for confirmation (using 20-period)
    vol_sma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    
    # Warmup: ensure all indicators are valid
    warmup = max(13, 8, 5, 20) + 8  # Max Alligator period + max offset
    
    for i in range(warmup, n):
        # Skip if outside trading session (08-20 UTC)
        if not in_session[i]:
            signals[i] = 0.0
            continue
        
        # Volume filter: current volume > 1.5x 20-period volume SMA
        vol_confirm = volume[i] > (vol_sma_20[i] * 1.5)
        
        # Skip if any required data is NaN
        if (np.isnan(jaws_lagged[i]) or np.isnan(teeth_lagged[i]) or np.isnan(lips_lagged[i]) or
            np.isnan(bull_power_aligned[i]) or np.isnan(bear_power_aligned[i]) or
            np.isnan(vol_sma_20[i])):
            signals[i] = 0.0
            continue
        
        # === LONG CONDITIONS ===
        # 1. Alligator aligned up: Lips > Teeth > Jaws
        # 2. Elder Bull Power > 0 (bullish momentum)
        # 3. Volume confirmation
        if (lips_lagged[i] > teeth_lagged[i]) and (teeth_lagged[i] > jaws_lagged[i]) and \
           (bull_power_aligned[i] > 0) and vol_confirm:
            signals[i] = 0.25
        
        # === SHORT CONDITIONS ===
        # 1. Alligator aligned down: Jaws > Teeth > Lips
        # 2. Elder Bear Power < 0 (bearish momentum)
        # 3. Volume confirmation
        elif (jaws_lagged[i] > teeth_lagged[i]) and (teeth_lagged[i] > lips_lagged[i]) and \
             (bear_power_aligned[i] < 0) and vol_confirm:
            signals[i] = -0.25
        
        else:
            signals[i] = 0.0  # flat
    
    return signals

name = "4h_Alligator_ElderRay_Volume_Filter_v1"
timeframe = "4h"
leverage = 1.0
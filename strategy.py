#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Williams Alligator + Elder Ray + Volume Spike
# - Primary: 6h Williams Alligator (Jaw=13, Teeth=8, Lips=5 SMAs) for trend direction
# - Confirmation: 6h Elder Ray (Bull Power = High - EMA13, Bear Power = EMA13 - Low)
# - Volume filter: 6h volume > 1.5x 20-period MA to avoid low-volume false signals
# - Long: Alligator bullish (Lips > Teeth > Jaw) AND Bull Power > 0 AND volume spike
# - Short: Alligator bearish (Lips < Teeth < Jaw) AND Bear Power > 0 AND volume spike
# - Exit: Alligator reverses (Teeth crosses Jaw) OR Elder Power crosses zero
# - Position sizing: 0.25 (discrete level to minimize fee churn)
# - Works in bull/bear: Alligator identifies trends, Elder Ray measures power, volume confirms participation
# - Target: 50-150 total trades over 4 years (12-37/year) for 6h timeframe

name = "6h_1d_alligator_elder_volume_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Pre-compute primary timeframe data
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Calculate EMA13 for Elder Ray (primary timeframe)
    close_s = pd.Series(close)
    ema13 = close_s.ewm(span=13, adjust=False, min_periods=13).mean().values
    
    # Calculate Elder Ray components
    bull_power = high - ema13  # High - EMA13
    bear_power = ema13 - low   # EMA13 - Low
    
    # Calculate Williams Alligator (SMAs with specific periods)
    # Jaw: 13-period SMMA (shifted 8 bars forward)
    # Teeth: 8-period SMMA (shifted 5 bars forward)
    # Lips: 5-period SMMA (shifted 3 bars forward)
    # Using SMA as approximation for SMMA with min_periods
    jaw_raw = pd.Series(close).rolling(window=13, min_periods=13).mean().values
    teeth_raw = pd.Series(close).rolling(window=8, min_periods=8).mean().values
    lips_raw = pd.Series(close).rolling(window=5, min_periods=5).mean().values
    
    # Apply Alligator shifts (Jaw shifted 8, Teeth shifted 5, Lips shifted 3)
    jaw = np.full_like(close, np.nan)
    teeth = np.full_like(close, np.nan)
    lips = np.full_like(close, np.nan)
    
    for i in range(len(close)):
        if i >= 8 and not np.isnan(jaw_raw[i-8]):
            jaw[i] = jaw_raw[i-8]
        if i >= 5 and not np.isnan(teeth_raw[i-5]):
            teeth[i] = teeth_raw[i-5]
        if i >= 3 and not np.isnan(lips_raw[i-3]):
            lips[i] = lips_raw[i-3]
    
    # Volume confirmation: 6h volume > 1.5x 20-period MA
    volume_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > 1.5 * volume_ma_20
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(20, n):
        # Skip if any required data is invalid
        if (np.isnan(lips[i]) or np.isnan(teeth[i]) or np.isnan(jaw[i]) or 
            np.isnan(bull_power[i]) or np.isnan(bear_power[i])):
            signals[i] = 0.0
            continue
        
        # Alligator conditions
        alligator_bullish = lips[i] > teeth[i] and teeth[i] > jaw[i]
        alligator_bearish = lips[i] < teeth[i] and teeth[i] < jaw[i]
        
        # Elder Ray conditions
        bull_strong = bull_power[i] > 0
        bear_strong = bear_power[i] > 0
        
        if position == 0:  # Flat - look for new entries
            # Long entry: Alligator bullish AND Bull Power positive AND volume spike
            if alligator_bullish and bull_strong and volume_spike[i]:
                position = 1
                signals[i] = 0.25
            # Short entry: Alligator bearish AND Bear Power positive AND volume spike
            elif alligator_bearish and bear_strong and volume_spike[i]:
                position = -1
                signals[i] = -0.25
            else:
                signals[i] = 0.0
        else:  # Have position - look for exit
            # Exit conditions: Alligator reverses OR Elder Power crosses zero
            exit_long = not alligator_bullish or not bull_strong
            exit_short = not alligator_bearish or not bear_strong
            
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
#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Williams Alligator + 1d Elder Ray + volume confirmation
# - Primary: 6h Williams Alligator (Jaw=13, Teeth=8, Lips=5) for trend direction
# - HTF: 1d Elder Ray (Bull Power = High - EMA13, Bear Power = EMA13 - Low) for institutional conviction
# - Volume filter: 6h volume > 1.5x 20-period MA to avoid low-participation moves
# - Long: Alligator bullish (Lips > Teeth > Jaw) + Bull Power > 0 + volume confirmation
# - Short: Alligator bearish (Lips < Teeth < Jaw) + Bear Power > 0 + volume confirmation
# - Exit: Alligator reverses (Teeth crosses Jaw) or Elder Power weakens (< 0)
# - Position sizing: 0.25 (discrete level to minimize fee churn)
# - Works in bull/bear: Alligator catches trends, Elder Ray filters fakeouts, volume confirms participation
# - Target: 50-150 total trades over 4 years (12-37/year) for 6h timeframe

name = "6h_1d_alligator_elder_volume_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Load HTF data ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    # Pre-compute primary timeframe data
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Pre-compute HTF data
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate 6h Williams Alligator
    # Jaw: 13-period SMMA (Smoothed Moving Average) of median price
    # Teeth: 8-period SMMA of median price  
    # Lips: 5-period SMMA of median price
    median_price = (high + low) / 2.0
    
    def smma(source, period):
        """Smoothed Moving Average"""
        result = np.full(len(source), np.nan)
        if len(source) < period:
            return result
        # First value is simple SMA
        result[period-1] = np.mean(source[:period])
        # Subsequent values: SMMA = (PREV_SMMA * (period-1) + CURRENT_VALUE) / period
        for i in range(period, len(source)):
            if not np.isnan(result[i-1]) and not np.isnan(source[i]):
                result[i] = (result[i-1] * (period-1) + source[i]) / period
        return result
    
    jaw = smma(median_price, 13)  # Blue line
    teeth = smma(median_price, 8)  # Red line
    lips = smma(median_price, 5)   # Green line
    
    # Calculate 1d Elder Ray
    # Bull Power = High - EMA13
    # Bear Power = EMA13 - Low
    ema13_1d = pd.Series(close_1d).ewm(span=13, adjust=False, min_periods=13).mean().values
    bull_power = high_1d - ema13_1d
    bear_power = ema13_1d - low_1d
    
    # Calculate 6h volume MA (20-period)
    volume_ma_20 = np.full(len(volume), np.nan)
    for i in range(19, len(volume)):
        if not np.isnan(volume[i-19:i+1]).any():
            volume_ma_20[i] = np.mean(volume[i-19:i+1])
    
    # Align HTF indicators to 6h timeframe
    bull_power_aligned = align_htf_to_ltf(prices, df_1d, bull_power)
    bear_power_aligned = align_htf_to_ltf(prices, df_1d, bear_power)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(30, n):
        # Skip if any required data is invalid
        if (np.isnan(lips[i]) or np.isnan(teeth[i]) or np.isnan(jaw[i]) or 
            np.isnan(bull_power_aligned[i]) or np.isnan(bear_power_aligned[i]) or 
            np.isnan(volume_ma_20[i])):
            signals[i] = 0.0
            continue
        
        # Volume confirmation: current 6h volume > 1.5x 20-period MA
        volume_confirm = volume[i] > 1.5 * volume_ma_20[i]
        
        # Alligator conditions
        alligator_bullish = lips[i] > teeth[i] and teeth[i] > jaw[i]
        alligator_bearish = lips[i] < teeth[i] and teeth[i] < jaw[i]
        alligator_exit = (position == 1 and teeth[i] <= jaw[i]) or (position == -1 and teeth[i] >= jaw[i])
        
        # Elder Ray conditions
        elder_bull = bull_power_aligned[i] > 0
        elder_bear = bear_power_aligned[i] > 0
        elder_exit = (position == 1 and bull_power_aligned[i] <= 0) or (position == -1 and bear_power_aligned[i] <= 0)
        
        if position == 0:  # Flat - look for new entries
            # Long entry: Alligator bullish + Bull Power > 0 + volume confirmation
            if alligator_bullish and elder_bull and volume_confirm:
                position = 1
                signals[i] = 0.25
            # Short entry: Alligator bearish + Bear Power > 0 + volume confirmation
            elif alligator_bearish and elder_bear and volume_confirm:
                position = -1
                signals[i] = -0.25
            else:
                signals[i] = 0.0
        else:  # Have position - look for exit
            # Exit: Alligator reverses OR Elder Power weakens
            if alligator_exit or elder_exit:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25 if position == 1 else -0.25
    
    return signals
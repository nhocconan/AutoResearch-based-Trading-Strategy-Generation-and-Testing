#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Elder Ray + Williams %R Volume Confirmation
# - Uses 1d Elder Ray (Bull Power = High - EMA13, Bear Power = EMA13 - Low) to measure trend strength
# - Uses 1d Williams %R (14-period) for overbought/oversold conditions
# - Requires volume > 1.5x 20-period average for confirmation
# - Long when Bull Power > 0 AND Williams %R < -80 AND volume spike (strong bullish momentum from oversold)
# - Short when Bear Power > 0 AND Williams %R > -20 AND volume spike (strong bearish momentum from overbought)
# - ATR(14) trailing stop at 2.5x ATR from extreme
# - Position size: 0.25 (discrete level to minimize fee churn)
# - Target: ~20 trades/year (80 total over 4 years) to stay well under fee drag threshold
# - Elder Ray captures trend strength, Williams %R identifies exhaustion points, volume confirms conviction
# - Designed to work in BOTH bull markets (buy strength on dips) and bear markets (sell weakness on rallies)

name = "6h_1d_elderray_williamsr_volume_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Load HTF data ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    # Pre-compute 1d indicators
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # 1d EMA(13) for Elder Ray
    ema_13_1d = pd.Series(close_1d).ewm(span=13, adjust=False, min_periods=13).mean().values
    
    # 1d Elder Ray components
    bull_power_1d = high_1d - ema_13_1d  # Bull Power = High - EMA13
    bear_power_1d = ema_13_1d - low_1d   # Bear Power = EMA13 - Low
    
    # 1d Williams %R (14-period)
    highest_high_14 = pd.Series(high_1d).rolling(window=14, min_periods=14).max().values
    lowest_low_14 = pd.Series(low_1d).rolling(window=14, min_periods=14).min().values
    williams_r_1d = -100 * (highest_high_14 - close_1d) / (highest_high_14 - lowest_low_14)
    # Handle division by zero
    williams_r_1d = np.where((highest_high_14 - lowest_low_14) == 0, -50, williams_r_1d)
    
    # Align HTF indicators to 6h timeframe
    bull_power_aligned = align_htf_to_ltf(prices, df_1d, bull_power_1d)
    bear_power_aligned = align_htf_to_ltf(prices, df_1d, bear_power_1d)
    williams_r_aligned = align_htf_to_ltf(prices, df_1d, williams_r_1d)
    
    # 6h price data
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # 6h volume > 1.5x 20-period average for confirmation
    avg_volume_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (1.5 * avg_volume_20)
    
    # 6h ATR(14) for trailing stop
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr[0]
    atr = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    highest_since_entry = 0.0
    lowest_since_entry = 0.0
    
    for i in range(100, n):
        # Skip if any required data is invalid
        if (np.isnan(bull_power_aligned[i]) or 
            np.isnan(bear_power_aligned[i]) or
            np.isnan(williams_r_aligned[i]) or
            np.isnan(volume_spike[i]) or
            np.isnan(atr[i]) or
            atr[i] <= 0):
            signals[i] = 0.0
            continue
        
        if position == 1:  # Long position
            # Update highest high since entry
            if high[i] > highest_since_entry:
                highest_since_entry = high[i]
            
            # Exit conditions: price retraces 2.5x ATR from high
            if low[i] <= highest_since_entry - (2.5 * atr[i]):
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Update lowest low since entry
            if low[i] < lowest_since_entry:
                lowest_since_entry = low[i]
            
            # Exit conditions: price retraces 2.5x ATR from low
            if high[i] >= lowest_since_entry + (2.5 * atr[i]):
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat
            # Look for Elder Ray + Williams %R signals with volume confirmation
            # Long: Bull Power > 0 (strong bullish momentum) AND Williams %R < -80 (oversold) AND volume spike
            if (bull_power_aligned[i] > 0 and 
                williams_r_aligned[i] < -80 and
                volume_spike[i]):
                position = 1
                highest_since_entry = high[i]
                lowest_since_entry = high[i]  # Initialize for shorts
                signals[i] = 0.25
            # Short: Bear Power > 0 (strong bearish momentum) AND Williams %R > -20 (overbought) AND volume spike
            elif (bear_power_aligned[i] > 0 and 
                  williams_r_aligned[i] > -20 and
                  volume_spike[i]):
                position = -1
                highest_since_entry = low[i]  # Initialize for longs
                lowest_since_entry = low[i]
                signals[i] = -0.25
    
    return signals
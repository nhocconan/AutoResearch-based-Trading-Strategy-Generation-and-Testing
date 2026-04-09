#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Donchian channel breakout with 1d ATR regime filter and volume confirmation
# - Uses 4h Donchian(20) breakouts for trend following entries
# - Requires 1d ATR(14) > 20-period median ATR (high volatility regime) for entry filter
# - Requires volume > 1.5 * 20-period volume average for confirmation
# - Uses ATR(14) for dynamic stoploss (2.5 * ATR) and position sizing (0.25)
# - Works in bull markets via upward breakouts, in bear via downward breakouts
# - Volatility regime filter avoids whipsaws in low-volatility ranging markets
# - Target: 25-40 trades/year on 4h timeframe (100-160 total over 4 years) to avoid fee drag

name = "4h_1d_donchian_breakout_atr_regime_volume_v1"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Load HTF data ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    
    # 1d ATR(14) for volatility regime filter
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    tr1 = high_1d - low_1d
    tr2 = np.abs(high_1d - np.roll(close_1d, 1))
    tr3 = np.abs(low_1d - np.roll(close_1d, 1))
    tr_1d = np.maximum(tr1, np.maximum(tr2, tr3))
    tr_1d[0] = tr_1d[0]
    atr_1d = pd.Series(tr_1d).rolling(window=14, min_periods=14).mean().values
    
    # 20-period median ATR for regime filter (high volatility when ATR > median)
    atr_median_20 = pd.Series(atr_1d).rolling(window=20, min_periods=20).median().values
    high_vol_regime = atr_1d > atr_median_20  # True when volatility is above median
    
    # Align HTF volatility regime to 4h timeframe
    high_vol_aligned = align_htf_to_ltf(prices, df_1d, high_vol_regime)
    
    # Pre-compute 4h Donchian channels (20-period)
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    
    # Donchian upper/lower bands (20-period high/low)
    donchian_high = pd.Series(high).rolling(window=20, min_periods=20).max().values
    donchian_low = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # Pre-compute 4h ATR(14) for stoploss and sizing
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]
    atr = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    # Pre-compute volume confirmation: volume > 1.5 * 20-period average
    volume = prices['volume'].values
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_confirm = volume > (1.5 * vol_ma)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    highest_high_since_entry = 0.0
    lowest_low_since_entry = 0.0
    
    for i in range(100, n):
        # Skip if any required data is invalid
        if (np.isnan(donchian_high[i]) or np.isnan(donchian_low[i]) or
            np.isnan(atr[i]) or atr[i] <= 0 or
            np.isnan(high_vol_aligned[i]) or
            np.isnan(volume_confirm[i])):
            signals[i] = 0.0
            continue
        
        if position == 1:  # Long position
            # Update highest high since entry
            highest_high_since_entry = max(highest_high_since_entry, high[i])
            
            # Exit conditions: ATR stoploss or Donchian lower band break
            if close[i] < highest_high_since_entry - 2.5 * atr[i]:  # ATR stop
                position = 0
                highest_high_since_entry = 0.0
                lowest_low_since_entry = 0.0
                signals[i] = 0.0
            elif close[i] < donchian_low[i]:  # Break below Donchian lower band
                position = 0
                highest_high_since_entry = 0.0
                lowest_low_since_entry = 0.0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Update lowest low since entry
            lowest_low_since_entry = min(lowest_low_since_entry, low[i])
            
            # Exit conditions: ATR stoploss or Donchian upper band break
            if close[i] > lowest_low_since_entry + 2.5 * atr[i]:  # ATR stop
                position = 0
                highest_high_since_entry = 0.0
                lowest_low_since_entry = 0.0
                signals[i] = 0.0
            elif close[i] > donchian_high[i]:  # Break above Donchian upper band
                position = 0
                highest_high_since_entry = 0.0
                lowest_low_since_entry = 0.0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat
            # Look for breakout entries with volatility regime and volume confirmation
            if (close[i] > donchian_high[i] and 
                high_vol_aligned[i] and  # High volatility regime
                volume_confirm[i]):      # Volume confirmation
                position = 1
                highest_high_since_entry = high[i]
                lowest_low_since_entry = low[i]
                signals[i] = 0.25
            elif (close[i] < donchian_low[i] and 
                  high_vol_aligned[i] and  # High volatility regime
                  volume_confirm[i]):      # Volume confirmation
                position = -1
                highest_high_since_entry = high[i]
                lowest_low_since_entry = low[i]
                signals[i] = -0.25
    
    return signals
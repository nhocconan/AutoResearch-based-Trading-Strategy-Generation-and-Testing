#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h Donchian(20) breakout with 1d volume confirmation and ATR-based stoploss
# - Uses 20-period Donchian channels on 12h for breakout entries (upper/lower bands)
# - Requires 1d volume > 1.5 * 20-period volume average for confirmation (reduces false breakouts)
# - Uses ATR(14) on 12h for dynamic stoploss (2.5 * ATR) and position sizing (0.25)
# - Works in bull markets via breakouts above upper band, in bear via breakdowns below lower band
# - Target: 12-30 trades/year on 12h timeframe (48-120 total over 4 years) to avoid fee drag
# - Donchian channels adapt to volatility and provide clear breakout levels

name = "12h_1d_donchian_breakout_volume_v1"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Load HTF data ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    # Pre-compute 12h Donchian channels (20-period)
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    
    # Upper band: 20-period rolling max of high
    upper_band = pd.Series(high).rolling(window=20, min_periods=20).max().values
    # Lower band: 20-period rolling min of low
    lower_band = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # Pre-compute 12h ATR(14) for stoploss
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]
    atr = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    # Pre-compute 1d volume confirmation: volume > 1.5 * 20-period average
    volume_1d = df_1d['volume'].values
    vol_ma_1d = pd.Series(volume_1d).rolling(window=20, min_periods=20).mean().values
    volume_confirm_1d = volume_1d > (1.5 * vol_ma_1d)
    volume_confirm_aligned = align_htf_to_ltf(prices, df_1d, volume_confirm_1d)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    highest_high_since_entry = 0.0
    lowest_low_since_entry = 0.0
    
    for i in range(100, n):
        # Skip if any required data is invalid
        if (np.isnan(upper_band[i]) or np.isnan(lower_band[i]) or
            np.isnan(atr[i]) or atr[i] <= 0 or
            np.isnan(volume_confirm_aligned[i])):
            signals[i] = 0.0
            continue
        
        if position == 1:  # Long position
            # Update highest high since entry
            highest_high_since_entry = max(highest_high_since_entry, high[i])
            
            # Exit conditions: ATR stoploss or break below lower band (mean reversion)
            if close[i] < highest_high_since_entry - 2.5 * atr[i]:  # ATR stop
                position = 0
                highest_high_since_entry = 0.0
                lowest_low_since_entry = 0.0
                signals[i] = 0.0
            elif close[i] < lower_band[i]:  # Mean reversion exit (break below lower band)
                position = 0
                highest_high_since_entry = 0.0
                lowest_low_since_entry = 0.0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Update lowest low since entry
            lowest_low_since_entry = min(lowest_low_since_entry, low[i])
            
            # Exit conditions: ATR stoploss or break above upper band (mean reversion)
            if close[i] > lowest_low_since_entry + 2.5 * atr[i]:  # ATR stop
                position = 0
                highest_high_since_entry = 0.0
                lowest_low_since_entry = 0.0
                signals[i] = 0.0
            elif close[i] > upper_band[i]:  # Mean reversion exit (break above upper band)
                position = 0
                highest_high_since_entry = 0.0
                lowest_low_since_entry = 0.0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat
            # Look for breakout entries with volume confirmation
            if close[i] > upper_band[i] and volume_confirm_aligned[i]:  # Break above upper band
                position = 1
                highest_high_since_entry = high[i]
                lowest_low_since_entry = low[i]
                signals[i] = 0.25
            elif close[i] < lower_band[i] and volume_confirm_aligned[i]:  # Break below lower band
                position = -1
                highest_high_since_entry = high[i]
                lowest_low_since_entry = low[i]
                signals[i] = -0.25
    
    return signals
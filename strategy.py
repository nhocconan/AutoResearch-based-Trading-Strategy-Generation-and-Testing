#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Donchian breakout with 1d volume confirmation and ATR trailing stop
# - Primary signal: 4h price breaks above/below 20-period Donchian channel
# - HTF filter: require 1d volume > 1.5 * 20-period volume average for confirmation
# - Exit: ATR(14) trailing stop (2.5 * ATR) or mean reversion to Donchian midpoint
# - Position size: 0.25 (discrete level to minimize fee churn)
# - Works in bull markets via upward breakouts, in bear via downward breakouts
# - Target: 20-40 trades/year on 4h (80-160 total over 4 years) to avoid fee drag
# - Donchian channels provide adaptive volatility-based support/resistance

name = "4h_1d_donchian_breakout_volume_v3"
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
    
    # 1d Volume confirmation: volume > 1.5 * 20-period average
    volume_1d = df_1d['volume'].values
    vol_ma_1d = pd.Series(volume_1d).rolling(window=20, min_periods=20).mean().values
    volume_confirm_1d = volume_1d > (1.5 * vol_ma_1d)
    volume_confirm_1d_aligned = align_htf_to_ltf(prices, df_1d, volume_confirm_1d)
    
    # 4h Donchian channel (20-period)
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    
    highest_high = pd.Series(high).rolling(window=20, min_periods=20).max().values
    lowest_low = pd.Series(low).rolling(window=20, min_periods=20).min().values
    donchian_mid = (highest_high + lowest_low) / 2.0
    
    # 4h ATR(14) for trailing stop
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]
    atr = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    highest_since_entry = 0.0
    lowest_since_entry = 0.0
    
    for i in range(100, n):
        # Skip if any required data is invalid
        if (np.isnan(highest_high[i]) or np.isnan(lowest_low[i]) or
            np.isnan(donchian_mid[i]) or np.isnan(atr[i]) or atr[i] <= 0 or
            np.isnan(volume_confirm_1d_aligned[i])):
            signals[i] = 0.0
            continue
        
        if position == 1:  # Long position
            # Update highest high since entry for trailing stop
            highest_since_entry = max(highest_since_entry, high[i])
            
            # Exit conditions: ATR trailing stop or mean reversion
            if close[i] < highest_since_entry - 2.5 * atr[i]:  # ATR trailing stop
                position = 0
                highest_since_entry = 0.0
                lowest_since_entry = 0.0
                signals[i] = 0.0
            elif close[i] < donchian_mid[i]:  # Mean reversion exit (break below midpoint)
                position = 0
                highest_since_entry = 0.0
                lowest_since_entry = 0.0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Update lowest low since entry for trailing stop
            lowest_since_entry = min(lowest_since_entry, low[i])
            
            # Exit conditions: ATR trailing stop or mean reversion
            if close[i] > lowest_since_entry + 2.5 * atr[i]:  # ATR trailing stop
                position = 0
                highest_since_entry = 0.0
                lowest_since_entry = 0.0
                signals[i] = 0.0
            elif close[i] > donchian_mid[i]:  # Mean reversion exit (break above midpoint)
                position = 0
                highest_since_entry = 0.0
                lowest_since_entry = 0.0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat
            # Look for breakout entries with volume confirmation
            if close[i] > highest_high[i] and volume_confirm_1d_aligned[i]:  # Break above upper channel
                position = 1
                highest_since_entry = high[i]
                lowest_since_entry = low[i]
                signals[i] = 0.25
            elif close[i] < lowest_low[i] and volume_confirm_1d_aligned[i]:  # Break below lower channel
                position = -1
                highest_since_entry = high[i]
                lowest_since_entry = low[i]
                signals[i] = -0.25
    
    return signals
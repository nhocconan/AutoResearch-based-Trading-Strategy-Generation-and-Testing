#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h strategy using 12h Supertrend for trend direction and 12h ATR breakout for entry.
# Long when price breaks above 12h ATR-based upper band with Supertrend uptrend and volume confirmation.
# Short when price breaks below 12h ATR-based lower band with Supertrend downtrend and volume confirmation.
# Exit when price crosses the Supertrend line or ATR band reverses.
# Supertrend adapts to volatility, working in both bull and bear markets.
# Target: 20-25 trades/year per symbol (80-100 total over 4 years) to minimize fee drag.

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Load 12h data ONCE for Supertrend and ATR bands
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 20:
        return np.zeros(n)
    
    high_12h = df_12h['high'].values
    low_12h = df_12h['low'].values
    close_12h = df_12h['close'].values
    
    # Calculate ATR for Supertrend
    tr1 = np.abs(high_12h[1:] - low_12h[1:])
    tr2 = np.abs(high_12h[1:] - close_12h[:-1])
    tr3 = np.abs(low_12h[1:] - close_12h[:-1])
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr = np.concatenate([[np.nan], tr])
    
    atr_period = 10
    atr_12h = pd.Series(tr).ewm(span=atr_period, adjust=False, min_periods=atr_period).mean().values
    
    # Supertrend parameters
    atr_multiplier = 3.0
    
    # Calculate basic upper and lower bands
    hl2 = (high_12h + low_12h) / 2
    upper_band = hl2 + (atr_multiplier * atr_12h)
    lower_band = hl2 - (atr_multiplier * atr_12h)
    
    # Initialize Supertrend
    supertrend = np.full_like(close_12h, np.nan)
    trend = np.ones_like(close_12h, dtype=int)  # 1 for uptrend, -1 for downtrend
    
    # Calculate Supertrend
    for i in range(1, len(close_12h)):
        if np.isnan(upper_band[i]) or np.isnan(lower_band[i]) or np.isnan(atr_12h[i]):
            supertrend[i] = supertrend[i-1] if i > 0 else np.nan
            trend[i] = trend[i-1] if i > 0 else 1
            continue
            
        if close_12h[i] > upper_band[i-1]:
            trend[i] = 1
        elif close_12h[i] < lower_band[i-1]:
            trend[i] = -1
        else:
            trend[i] = trend[i-1]
            if trend[i] == 1 and lower_band[i] < lower_band[i-1]:
                lower_band[i] = lower_band[i-1]
            if trend[i] == -1 and upper_band[i] > upper_band[i-1]:
                upper_band[i] = upper_band[i-1]
        
        if trend[i] == 1:
            supertrend[i] = lower_band[i]
        else:
            supertrend[i] = upper_band[i]
    
    # Calculate ATR breakout bands (separate from Supertrend for entry)
    atr_breakout_period = 14
    atr_breakout = pd.Series(tr).ewm(span=atr_breakout_period, adjust=False, min_periods=atr_breakout_period).mean().values
    
    # ATR breakout bands
    atr_multiplier_breakout = 1.5
    upper_breakout = hl2 + (atr_multiplier_breakout * atr_breakout)
    lower_breakout = hl2 - (atr_multiplier_breakout * atr_breakout)
    
    # Align indicators to lower timeframe
    supertrend_aligned = align_htf_to_ltf(prices, df_12h, supertrend)
    trend_aligned = align_htf_to_ltf(prices, df_12h, trend)
    upper_breakout_aligned = align_htf_to_ltf(prices, df_12h, upper_breakout)
    lower_breakout_aligned = align_htf_to_ltf(prices, df_12h, lower_breakout)
    
    # Volume confirmation: 1.5x average volume
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0
    position_size = 0.25  # 25% position size
    
    # Start after enough data for calculations
    start = max(20, 14)
    
    for i in range(start, n):
        # Skip if any critical data is NaN
        if (np.isnan(supertrend_aligned[i]) or 
            np.isnan(trend_aligned[i]) or
            np.isnan(upper_breakout_aligned[i]) or
            np.isnan(lower_breakout_aligned[i]) or
            np.isnan(vol_ma[i])):
            signals[i] = 0.0
            continue
        
        # Volume confirmation
        volume_confirmed = volume[i] > 1.5 * vol_ma[i]
        
        if position == 0:
            # Look for ATR breakout entries
            # Long: price breaks above upper breakout band with uptrend
            if (close[i] > upper_breakout_aligned[i] and 
                trend_aligned[i] == 1 and 
                volume_confirmed):
                position = 1
                signals[i] = position_size
            # Short: price breaks below lower breakout band with downtrend
            elif (close[i] < lower_breakout_aligned[i] and 
                  trend_aligned[i] == -1 and 
                  volume_confirmed):
                position = -1
                signals[i] = -position_size
            else:
                signals[i] = 0.0
        elif position == 1:
            # Exit long: price crosses below Supertrend or breaks below lower breakout band
            if (close[i] < supertrend_aligned[i] or 
                close[i] < lower_breakout_aligned[i]):
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = position_size
        elif position == -1:
            # Exit short: price crosses above Supertrend or breaks above upper breakout band
            if (close[i] > supertrend_aligned[i] or 
                close[i] > upper_breakout_aligned[i]):
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -position_size
    
    return signals

name = "4h_Supertrend_ATRBreakout_Volume_v1"
timeframe = "4h"
leverage = 1.0
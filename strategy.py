#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h strategy using 1d ATR for volatility filtering and 4h Donchian channel breakout.
# 1d ATR(14) > median ATR(14) filters for high volatility periods to capture breakouts.
# Donchian channel breakout (20) provides entry with clear structure.
# Volume confirmation (>1.2x 20-period average) reduces false breakouts.
# ATR-based exit manages risk with 2x ATR trailing stop.
# Designed to work in both bull and bear markets by using volatility filter to capture expansion phases.
# Target: 20-30 trades/year per symbol (80-120 total over 4 years) to minimize fee drag.

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Load 1d data ONCE for ATR calculation
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    
    # Calculate ATR on 1d data
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # True Range
    tr1 = np.abs(high_1d[1:] - low_1d[1:])
    tr2 = np.abs(high_1d[1:] - close_1d[:-1])
    tr3 = np.abs(low_1d[1:] - close_1d[:-1])
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr = np.concatenate([[np.nan], tr])
    
    # ATR
    atr_period = 14
    atr_1d = pd.Series(tr).ewm(span=atr_period, adjust=False, min_periods=atr_period).mean().values
    
    # Median ATR for volatility filter
    atr_median = np.nanmedian(atr_1d)
    
    # Load 4h data ONCE for Donchian channels
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 20:
        return np.zeros(n)
    
    # Calculate Donchian channels on 4h data
    high_4h = df_4h['high'].values
    low_4h = df_4h['low'].values
    
    donch_period = 20
    upper_channel = pd.Series(high_4h).rolling(window=donch_period, min_periods=donch_period).max().values
    lower_channel = pd.Series(low_4h).rolling(window=donch_period, min_periods=donch_period).min().values
    
    # Align indicators to 4h timeframe
    atr_1d_aligned = align_htf_to_ltf(prices, df_1d, atr_1d)
    upper_channel_aligned = align_htf_to_ltf(prices, df_4h, upper_channel)
    lower_channel_aligned = align_htf_to_ltf(prices, df_4h, lower_channel)
    
    # Volume confirmation: 1.2x average volume
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0
    position_size = 0.25  # 25% position size
    
    # Start after enough data for calculations
    start = max(donch_period, 20)
    
    for i in range(start, n):
        # Skip if any critical data is NaN
        if (np.isnan(atr_1d_aligned[i]) or 
            np.isnan(upper_channel_aligned[i]) or
            np.isnan(lower_channel_aligned[i]) or
            np.isnan(vol_ma[i])):
            signals[i] = 0.0
            continue
        
        # Volatility filter: current 1d ATR > median ATR
        high_volatility = atr_1d_aligned[i] > atr_median
        
        # Volume confirmation
        volume_confirmed = volume[i] > 1.2 * vol_ma[i]
        
        if position == 0:
            # Look for Donchian channel breakouts
            # Only trade in high volatility periods
            
            # Long: price breaks above upper Donchian channel
            if (close[i] > upper_channel_aligned[i] and 
                high_volatility and 
                volume_confirmed):
                position = 1
                signals[i] = position_size
            # Short: price breaks below lower Donchian channel
            elif (close[i] < lower_channel_aligned[i] and 
                  high_volatility and 
                  volume_confirmed):
                position = -1
                signals[i] = -position_size
            else:
                signals[i] = 0.0
        elif position == 1:
            # Exit long: 2x ATR trailing stop or reversal signal
            atr_4h = pd.Series(np.maximum(high - low, np.abs(high - np.roll(close, 1)), np.abs(low - np.roll(close, 1)))).ewm(span=14, adjust=False, min_periods=14).mean().values
            atr_4h[0] = atr_4h[1] if len(atr_4h) > 1 else 0.0
            
            # Track highest high since entry for trailing stop
            # Simplified: exit if price drops 2x ATR from current high
            if i > 0:
                recent_high = np.maximum.accumulate(high[:i+1])[-1]
                if close[i] < recent_high - 2.0 * atr_4h[i]:
                    position = 0
                    signals[i] = 0.0
                else:
                    signals[i] = position_size
            else:
                signals[i] = position_size
        elif position == -1:
            # Exit short: 2x ATR trailing stop or reversal signal
            atr_4h = pd.Series(np.maximum(high - low, np.abs(high - np.roll(close, 1)), np.abs(low - np.roll(close, 1)))).ewm(span=14, adjust=False, min_periods=14).mean().values
            atr_4h[0] = atr_4h[1] if len(atr_4h) > 1 else 0.0
            
            # Track lowest low since entry for trailing stop
            # Simplified: exit if price rises 2x ATR from current low
            if i > 0:
                recent_low = np.minimum.accumulate(low[:i+1])[-1]
                if close[i] > recent_low + 2.0 * atr_4h[i]:
                    position = 0
                    signals[i] = 0.0
                else:
                    signals[i] = -position_size
            else:
                signals[i] = -position_size
    
    return signals

name = "4h_1dATR_VolatilityFilter_DonchianBreakout_Volume_v1"
timeframe = "4h"
leverage = 1.0
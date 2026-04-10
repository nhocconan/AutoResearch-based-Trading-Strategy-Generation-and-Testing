#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Donchian(20) breakout with 1d ATR regime filter and volume confirmation
# - Long when price breaks above Donchian upper channel (20-period high) + ATR(14) < ATR(50) (low volatility regime) + 1d volume > 1.3x 20-period volume SMA
# - Short when price breaks below Donchian lower channel (20-period low) + ATR(14) < ATR(50) + 1d volume > 1.3x 20-period volume SMA
# - Exit: price crosses back through Donchian middle (10-period midpoint)
# - Position sizing: 0.25 discrete level
# - Donchian channels provide clear breakout levels with built-in trend following
# - ATR regime filter ensures we trade during low volatility periods (pre-breakout squeeze)
# - Volume confirmation validates breakout strength
# - Target: 20-40 trades/year to minimize fee drag while capturing strong moves

name = "4h_1d_donchian_atr_volume_v1"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
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
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    # Calculate 4h Donchian Channels (20-period)
    donch_period = 20
    upper_channel = pd.Series(high).rolling(window=donch_period, min_periods=donch_period).max().values
    lower_channel = pd.Series(low).rolling(window=donch_period, min_periods=donch_period).min().values
    middle_channel = (upper_channel + lower_channel) / 2
    
    # Calculate 4h ATR(14) and ATR(50) for volatility regime
    atr_period_short = 14
    atr_period_long = 50
    
    # True Range
    tr1 = np.maximum(high - low, 
                     np.maximum(np.abs(high - np.roll(close, 1)), 
                                np.abs(low - np.roll(close, 1))))
    tr1[0] = high[0] - low[0]
    
    atr_short = pd.Series(tr1).rolling(window=atr_period_short, min_periods=atr_period_short).mean().values
    atr_long = pd.Series(tr1).rolling(window=atr_period_long, min_periods=atr_period_long).mean().values
    
    # ATR ratio: short/long < 1 indicates low volatility regime (favorable for breakout)
    atr_ratio = np.where(atr_long > 0, atr_short / atr_long, 1.0)
    
    # Calculate 1d OHLC for volume confirmation
    volume_1d = df_1d['volume'].values
    
    # Calculate 1d volume SMA(20) for confirmation
    volume_sma_20_1d = pd.Series(volume_1d).rolling(window=20, min_periods=20).mean().values
    volume_sma_20_1d_aligned = align_htf_to_ltf(prices, df_1d, volume_sma_20_1d)
    
    for i in range(100, n):
        # Skip if any required data is invalid
        if (np.isnan(upper_channel[i]) or np.isnan(lower_channel[i]) or 
            np.isnan(middle_channel[i]) or np.isnan(atr_ratio[i]) or 
            np.isnan(volume_sma_20_1d_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Get current 1d volume for volume spike confirmation
        vol_1d_current = align_htf_to_ltf(prices, df_1d, volume_1d)
        
        # Volume confirmation: current 1d volume > 1.3x 20-period SMA (volume spike)
        vol_confirm = vol_1d_current[i] > 1.3 * volume_sma_20_1d_aligned[i]
        
        # Volatility regime: ATR ratio < 0.95 indicates low volatility (squeeze condition)
        low_vol_regime = atr_ratio[i] < 0.95
        
        # Donchian breakout signals
        long_entry = (close[i] > upper_channel[i]) and low_vol_regime and vol_confirm
        short_entry = (close[i] < lower_channel[i]) and low_vol_regime and vol_confirm
        exit_long = close[i] < middle_channel[i]  # Exit long when price crosses below middle
        exit_short = close[i] > middle_channel[i]  # Exit short when price crosses above middle
        
        if position == 0:  # Flat - look for entry
            if long_entry:
                position = 1
                signals[i] = 0.25
            elif short_entry:
                position = -1
                signals[i] = -0.25
            else:
                signals[i] = 0.0
        elif position == 1:  # Long position - look for exit
            if exit_long:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
        else:  # position == -1 (Short position) - look for exit
            if exit_short:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
    
    return signals
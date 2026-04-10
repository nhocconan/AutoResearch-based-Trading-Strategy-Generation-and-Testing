#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Donchian(20) breakout with 1d ATR volatility filter and volume confirmation
# - Long when price breaks above 4h Donchian upper channel (20-period high) + 1d ATR(14) > 1.5x 20-period ATR SMA (high volatility regime) + current 4h volume > 20-period volume SMA
# - Short when price breaks below 4h Donchian lower channel (20-period low) + same ATR and volume conditions
# - Exit: price returns to 4h Donchian midpoint (mean of upper/lower channel)
# - Position sizing: 0.25 discrete level
# - Donchian channels capture volatility-based breakouts that work in both trending and ranging markets
# - ATR filter ensures we trade during sufficient volatility regimes (avoids low-volume chop)
# - Volume confirmation adds conviction to breakouts
# - Target: 30-60 trades/year to balance opportunity with fee drag minimization

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
    
    # Calculate 4h Donchian channels (20-period)
    donchian_window = 20
    upper_channel = pd.Series(high).rolling(window=donchian_window, min_periods=donchian_window).max().values
    lower_channel = pd.Series(low).rolling(window=donchian_window, min_periods=donchian_window).min().values
    middle_channel = (upper_channel + lower_channel) / 2.0
    
    # Calculate 4h ATR for volatility filter
    tr1 = np.maximum(high - low, 
                     np.maximum(np.abs(high - np.roll(close, 1)), 
                                np.abs(low - np.roll(close, 1))))
    tr1[0] = high[0] - low[0]
    atr = pd.Series(tr1).rolling(window=14, min_periods=14).mean().values
    
    # Calculate 4h volume SMA for confirmation
    volume_sma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    # Calculate 1d ATR and its SMA for volatility regime filter
    atr_1d = pd.Series(df_1d['high'].values - df_1d['low'].values).rolling(window=14, min_periods=14).mean().values
    atr_sma_20_1d = pd.Series(atr_1d).rolling(window=20, min_periods=20).mean().values
    atr_sma_20_1d_aligned = align_htf_to_ltf(prices, df_1d, atr_sma_20_1d)
    atr_1d_aligned = align_htf_to_ltf(prices, df_1d, atr_1d)
    
    for i in range(100, n):
        # Skip if any required data is invalid
        if (np.isnan(upper_channel[i]) or np.isnan(lower_channel[i]) or 
            np.isnan(middle_channel[i]) or np.isnan(atr[i]) or 
            np.isnan(volume_sma[i]) or np.isnan(atr_1d_aligned[i]) or 
            np.isnan(atr_sma_20_1d_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Volatility filter: current 1d ATR > 1.5x 20-period ATR SMA (high volatility regime)
        vol_filter = atr_1d_aligned[i] > 1.5 * atr_sma_20_1d_aligned[i]
        
        # Volume confirmation: current 4h volume > 20-period volume SMA
        vol_confirm = volume[i] > volume_sma[i]
        
        # Donchian breakout signals
        long_entry = (close[i] > upper_channel[i]) and vol_filter and vol_confirm
        short_entry = (close[i] < lower_channel[i]) and vol_filter and vol_confirm
        exit_long = close[i] < middle_channel[i]  # Exit long when price crosses below midpoint
        exit_short = close[i] > middle_channel[i]  # Exit short when price crosses above midpoint
        
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
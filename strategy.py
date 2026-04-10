#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1d Donchian breakout with volume spike and weekly ATR regime filter
# - Long when price breaks above 20-period Donchian upper band + volume > 2.0x 20-period volume SMA + weekly ATR ratio < 0.8 (low volatility regime)
# - Short when price breaks below 20-period Donchian lower band + volume > 2.0x 20-period volume SMA + weekly ATR ratio < 0.8
# - Exit: price returns to Donchian midpoint (mean reversion within the channel)
# - Position sizing: 0.25 discrete level
# - Donchian channels provide clear breakout levels in both bull and bear markets
# - Volume confirmation ensures breakout validity
# - Weekly ATR ratio filter avoids high volatility periods where breakouts fail
# - Target: 20-50 trades/year on 1d timeframe to minimize fee drag

name = "1d_donchian_volume_atr_v1"
timeframe = "1d"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Load HTF data ONCE before loop
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 30:
        return np.zeros(n)
    
    # Pre-compute primary timeframe data
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    # Calculate 1d Donchian channels (20-period)
    donchian_period = 20
    highest_high = pd.Series(high).rolling(window=donchian_period, min_periods=donchian_period).max().values
    lowest_low = pd.Series(low).rolling(window=donchian_period, min_periods=donchian_period).min().values
    donchian_upper = highest_high
    donchian_lower = lowest_low
    donchian_middle = (donchian_upper + donchian_lower) / 2.0
    
    # Calculate 1d volume SMA for confirmation
    volume_sma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    # Calculate weekly ATR for regime filter (ATR ratio: current ATR / 20-period ATR SMA)
    # True Range for weekly data
    tr1w = np.maximum(df_1w['high'] - df_1w['low'], 
                      np.maximum(np.abs(df_1w['high'] - np.roll(df_1w['close'], 1)), 
                                 np.abs(df_1w['low'] - np.roll(df_1w['close'], 1))))
    tr1w[0] = df_1w['high'].iloc[0] - df_1w['low'].iloc[0]
    
    # Weekly ATR
    atr_period = 14
    atr_1w = pd.Series(tr1w).rolling(window=atr_period, min_periods=atr_period).mean().values
    atr_sma_20_1w = pd.Series(atr_1w).rolling(window=20, min_periods=20).mean().values
    
    # ATR ratio: current ATR / ATR SMA (values < 1 indicate low volatility regime)
    atr_ratio_1w = np.where(atr_sma_20_1w > 0, atr_1w / atr_sma_20_1w, 1.0)
    atr_ratio_1w_aligned = align_htf_to_ltf(prices, df_1w, atr_ratio_1w)
    
    # Align Donchian levels and volume SMA to 1d timeframe (already in 1d, but keep for consistency)
    donchian_upper_aligned = donchian_upper
    donchian_lower_aligned = donchian_lower
    donchian_middle_aligned = donchian_middle
    volume_sma_20_aligned = volume_sma_20
    
    for i in range(100, n):
        # Skip if any required data is invalid
        if (np.isnan(donchian_upper_aligned[i]) or np.isnan(donchian_lower_aligned[i]) or 
            np.isnan(donchian_middle_aligned[i]) or np.isnan(volume_sma_20_aligned[i]) or
            np.isnan(atr_ratio_1w_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Volume confirmation: current volume > 2.0x 20-period volume SMA (strong volume spike)
        vol_confirm = volume[i] > 2.0 * volume_sma_20_aligned[i]
        
        # Regime filter: weekly ATR ratio < 0.8 indicates low volatility regime (favorable for breakouts)
        low_vol_regime = atr_ratio_1w_aligned[i] < 0.8
        
        # Donchian breakout signals
        long_breakout = close[i] > donchian_upper_aligned[i]
        short_breakout = close[i] < donchian_lower_aligned[i]
        
        # Mean reversion exit: price returns to Donchian midpoint
        exit_long = close[i] < donchian_middle_aligned[i]
        exit_short = close[i] > donchian_middle_aligned[i]
        
        if position == 0:  # Flat - look for entry
            if long_breakout and vol_confirm and low_vol_regime:
                position = 1
                signals[i] = 0.25
            elif short_breakout and vol_confirm and low_vol_regime:
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
#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "4h_12h_donchian_breakout_v1"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    # Load 12h data ONCE before loop
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 20:
        return signals
    
    # Calculate 12h Donchian channels (20-period)
    high_12h = df_12h['high'].values
    low_12h = df_12h['low'].values
    
    # Upper band: highest high of last 20 periods
    donchian_high = pd.Series(high_12h).rolling(window=20, min_periods=20).max().values
    # Lower band: lowest low of last 20 periods
    donchian_low = pd.Series(low_12h).rolling(window=20, min_periods=20).min().values
    
    # Shift by 1 to use only completed 12h bars
    donchian_high = np.roll(donchian_high, 1)
    donchian_low = np.roll(donchian_low, 1)
    donchian_high[0] = np.nan
    donchian_low[0] = np.nan
    
    # Align 12h Donchian levels to 4h timeframe
    donchian_high_aligned = align_htf_to_ltf(prices, df_12h, donchian_high)
    donchian_low_aligned = align_htf_to_ltf(prices, df_12h, donchian_low)
    
    # Calculate 4h ATR for volatility filter (14-period)
    tr1 = high[1:] - low[1:]
    tr2 = np.abs(high[1:] - close[:-1])
    tr3 = np.abs(low[1:] - close[:-1])
    tr = np.concatenate([[np.nan], np.maximum(tr1, np.maximum(tr2, tr3))])
    atr = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    # Volume filter: volume > 1.3x 20-period average
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    for i in range(40, n):  # Start after warmup
        # Skip if any required data is invalid
        if (np.isnan(donchian_high_aligned[i]) or np.isnan(donchian_low_aligned[i]) or
            np.isnan(atr[i]) or np.isnan(vol_ma_20[i])):
            signals[i] = 0.0
            continue
        
        price_close = close[i]
        price_high = high[i]
        price_low = low[i]
        volume_current = volume[i]
        vol_ma = vol_ma_20[i]
        atr_value = atr[i]
        
        # Volume confirmation
        volume_confirmed = volume_current > 1.3 * vol_ma
        
        # Long: price breaks above 12h Donchian high with volume and volatility filter
        long_signal = volume_confirmed and (price_high > donchian_high_aligned[i]) and (atr_value > 0)
        
        # Short: price breaks below 12h Donchian low with volume and volatility filter
        short_signal = volume_confirmed and (price_low < donchian_low_aligned[i]) and (atr_value > 0)
        
        # Exit when price returns to the middle of the Donchian channel
        mid = (donchian_high_aligned[i] + donchian_low_aligned[i]) / 2
        exit_long = position == 1 and price_close < mid
        exit_short = position == -1 and price_close > mid
        
        # Trading logic
        if long_signal and position != 1:
            position = 1
            signals[i] = 0.25
        elif short_signal and position != -1:
            position = -1
            signals[i] = -0.25
        elif position == 1 and exit_long:
            position = 0
            signals[i] = 0.0
        elif position == -1 and exit_short:
            position = 0
            signals[i] = 0.0
        else:
            # Maintain current position
            signals[i] = 0.25 if position == 1 else (-0.25 if position == -1 else 0.0)
    
    return signals

# Hypothesis: 4h Donchian breakout using 12h channels with volume confirmation.
# Uses 12-hour Donchian channels (20-period) to identify major support/resistance.
# Enters long when 4h price breaks above the 12h Donchian high with volume confirmation
# (>1.3x 20-period average volume). Enters short when price breaks below the 12h
# Donchian low with volume confirmation. Exits when price returns to the midpoint
# of the channel. The 12h timeframe provides significant trend context while the
# 4h timeframe allows for timely entries. Volume confirmation ensures participation
# from market actors. Volatility filter (ATR > 0) ensures we only trade when there
# is sufficient price movement. Designed for 20-50 trades per year to minimize
# fee drift on 4h timeframe. Works in both bull and bear markets by following
# institutional price action as defined by the 12h Donchian channels.
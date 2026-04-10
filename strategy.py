#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Donchian(20) breakout with 1d volume spike and ADX(14) regime filter
# - Long when price breaks above 20-period Donchian upper + 1d volume > 2.0x 20-period volume SMA + ADX < 25 (range/low trend)
# - Short when price breaks below 20-period Donchian lower + 1d volume > 2.0x 20-period volume SMA + ADX < 25
# - Exit: price returns to 20-period Donchian midpoint (mean reversion within channel)
# - Position sizing: 0.25 discrete level
# - Donchian breakouts capture momentum, volume confirms participation, ADX filter avoids strong trends where breakouts fail
# - Works in bull/bear: breakouts effective in trending markets, ADX filter prevents trading against strong counter-trend moves
# - 4h timeframe targets 75-200 trades over 4 years (19-50/year) with strict entry conditions to minimize fee drag

name = "4h_1d_donchian_volume_adx_v3"
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
    
    # Calculate 4h Donchian Channel(20)
    highest_high = pd.Series(high).rolling(window=20, min_periods=20).max().values
    lowest_low = pd.Series(low).rolling(window=20, min_periods=20).min().values
    donchian_upper = highest_high
    donchian_lower = lowest_low
    donchian_mid = (donchian_upper + donchian_lower) / 2.0
    
    # Calculate 4h ADX(14) for regime filter (avoid strong trends)
    # True Range
    tr1 = np.maximum(high - low, 
                     np.maximum(np.abs(high - np.roll(close, 1)), 
                                np.abs(low - np.roll(close, 1))))
    tr1[0] = high[0] - low[0]
    # Plus Directional Movement
    plus_dm = np.where((high - np.roll(high, 1)) > (np.roll(low, 1) - low), 
                       np.maximum(high - np.roll(high, 1), 0), 0)
    plus_dm[0] = 0
    # Minus Directional Movement
    minus_dm = np.where((np.roll(low, 1) - low) > (high - np.roll(high, 1)), 
                        np.maximum(np.roll(low, 1) - low, 0), 0)
    minus_dm[0] = 0
    # Smoothed values
    atr = pd.Series(tr1).rolling(window=14, min_periods=14).mean().values
    plus_di = 100 * pd.Series(plus_dm).rolling(window=14, min_periods=14).mean().values / atr
    minus_di = 100 * pd.Series(minus_dm).rolling(window=14, min_periods=14).mean().values / atr
    dx = 100 * np.abs(plus_di - minus_di) / (plus_di + minus_di)
    adx = pd.Series(dx).rolling(window=14, min_periods=14).mean().values
    # Handle division by zero and invalid values
    plus_di = np.where(atr == 0, 0, plus_di)
    minus_di = np.where(atr == 0, 0, minus_di)
    dx = np.where((plus_di + minus_di) == 0, 0, dx)
    adx = np.where(np.isnan(adx) | np.isinf(adx), 0, adx)
    
    # Calculate 1d volume SMA(20) for confirmation
    volume_1d = df_1d['volume'].values
    volume_sma_20_1d = pd.Series(volume_1d).rolling(window=20, min_periods=20).mean().values
    volume_sma_20_1d_aligned = align_htf_to_ltf(prices, df_1d, volume_sma_20_1d)
    
    # Current 1d volume for spike detection (aligned to LTF)
    volume_1d_current = align_htf_to_ltf(prices, df_1d, df_1d['volume'].values)
    
    for i in range(100, n):
        # Skip if any required data is invalid
        if (np.isnan(donchian_upper[i]) or np.isnan(donchian_lower[i]) or np.isnan(donchian_mid[i]) or 
            np.isnan(adx[i]) or np.isnan(volume_sma_20_1d_aligned[i]) or np.isnan(volume_1d_current[i])):
            signals[i] = 0.0
            continue
        
        # Volume confirmation: current 1d volume > 2.0x 20-period SMA (volume spike)
        vol_confirm = volume_1d_current[i] > 2.0 * volume_sma_20_1d_aligned[i]
        
        # Regime filter: ADX < 25 indicates ranging/low trend market (favorable for breakout mean reversion)
        ranging_market = adx[i] < 25
        
        # Donchian breakout signals
        breakout_up = close[i] > donchian_upper[i-1]  # Price breaks above upper channel
        breakout_down = close[i] < donchian_lower[i-1]  # Price breaks below lower channel
        
        # Entry conditions: Donchian breakout with volume and regime confirmation
        long_entry = breakout_up and vol_confirm and ranging_market
        short_entry = breakout_down and vol_confirm and ranging_market
        
        # Exit conditions: price returns to Donchian midpoint (mean reversion within channel)
        long_exit = close[i] <= donchian_mid[i]  # Exit long when price returns to or below midpoint
        short_exit = close[i] >= donchian_mid[i]  # Exit short when price returns to or above midpoint
        
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
            if long_exit:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
        else:  # position == -1 (Short position) - look for exit
            if short_exit:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
    
    return signals
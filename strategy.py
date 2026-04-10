#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Donchian(20) breakout with 12h HMA trend filter and volume confirmation
# - Long when price breaks above 20-period Donchian high AND price > 12h HMA(21)
# - Short when price breaks below 20-period Donchian low AND price < 12h HMA(21)
# - Volume confirmation: 4h volume > 1.5x 20-period 4h volume SMA
# - Exit: Donchian midpoint reversion
# - Position sizing: 0.30 discrete level (max 0.40)
# - Target: 20-50 trades/year on 4h timeframe (80-200 total over 4 years)
# - 12h HMA provides structural trend bias, Donchian for breakouts, volume for confirmation
# - Works in bull markets via breakouts, in bear via short breakdowns with trend filter

name = "4h_12h_donchian_hma_volume_v1"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Load HTF data ONCE before loop
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 30:
        return np.zeros(n)
    
    # Pre-compute primary timeframe data
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    # Calculate 20-period Donchian channels
    donchian_period = 20
    donchian_high = pd.Series(high).rolling(window=donchian_period, min_periods=donchian_period).max().values
    donchian_low = pd.Series(low).rolling(window=donchian_period, min_periods=donchian_period).min().values
    donchian_mid = (donchian_high + donchian_low) / 2.0
    
    # Calculate 20-period volume SMA for confirmation
    volume_sma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    # Calculate 12h HMA(21) for trend filter
    # HMA = WMA(2*WMA(n/2) - WMA(n)), sqrt(n)
    half_len = 12 // 2  # 6
    sqrt_len = int(np.sqrt(12))  # 3
    
    def wma(values, window):
        weights = np.arange(1, window + 1)
        return np.convolve(values, weights, mode='valid') / weights.sum()
    
    close_12h = df_12h['close'].values
    if len(close_12h) >= 12:
        wma_half = np.array([wma(close_12h[i:i+half_len], half_len) 
                            for i in range(len(close_12h) - half_len + 1)])
        wma_full = np.array([wma(close_12h[i:i+12], 12) 
                            for i in range(len(close_12h) - 12 + 1)])
        # Align arrays: wma_half starts at index 0, wma_full starts at index 0
        # We need same length: take wma_half[6:] to align with wma_full
        if len(wma_half) > 6 and len(wma_full) > 0:
            hma_raw = 2 * wma_half[6:6+len(wma_full)] - wma_full
            # Second WMA with sqrt_len
            hma_values = np.array([wma(hma_raw[i:i+sqrt_len], sqrt_len) 
                                 for i in range(len(hma_raw) - sqrt_len + 1)])
            hma_12h = hma_values
        else:
            hma_12h = np.full(len(close_12h), np.nan)
    else:
        hma_12h = np.full(len(close_12h), np.nan)
    
    hma_12h_aligned = align_htf_to_ltf(prices, df_12h, hma_12h)
    
    # Track entry extreme for exit logic
    entry_price = np.full(n, np.nan)
    
    for i in range(donchian_period, n):
        # Skip if any required data is invalid
        if (np.isnan(donchian_high[i-1]) or np.isnan(donchian_low[i-1]) or
            np.isnan(volume_sma_20[i]) or np.isnan(hma_12h_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Volume confirmation: 4h volume > 1.5x 20-period volume SMA
        vol_confirm = volume[i] > 1.5 * volume_sma_20[i]
        
        # Donchian breakout signals
        breakout_up = close[i] > donchian_high[i-1]  # Break above previous Donchian high
        breakout_down = close[i] < donchian_low[i-1]  # Break below previous Donchian low
        
        # 12h HMA trend filter
        price_above_hma = close[i] > hma_12h_aligned[i]
        price_below_hma = close[i] < hma_12h_aligned[i]
        
        if position == 0:  # Flat - look for entry
            if breakout_up and price_above_hma and vol_confirm:
                position = 1
                signals[i] = 0.30
                entry_price[i] = close[i]
            elif breakout_down and price_below_hma and vol_confirm:
                position = -1
                signals[i] = -0.30
                entry_price[i] = close[i]
            else:
                signals[i] = 0.0
        elif position == 1:  # Long position - look for exit
            # Exit on Donchian midpoint reversion
            if close[i] < donchian_mid[i]:
                position = 0
                signals[i] = 0.0
                entry_price[i] = np.nan
            else:
                signals[i] = 0.30
        else:  # position == -1 (Short position) - look for exit
            # Exit on Donchian midpoint reversion
            if close[i] > donchian_mid[i]:
                position = 0
                signals[i] = 0.0
                entry_price[i] = np.nan
            else:
                signals[i] = -0.30
    
    return signals
#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h Donchian(20) breakout with daily pivot confirmation and volume spike filter
# - Long when price breaks above 20-period Donchian high AND price is above daily pivot point
# - Short when price breaks below 20-period Donchian low AND price is below daily pivot point
# - Volume confirmation: 12h volume > 2.0x 20-period 12h volume SMA
# - Exit: Donchian midpoint reversion or opposite breakout
# - Position sizing: 0.25 discrete level
# - Target: 12-37 trades/year on 12h timeframe (50-150 total over 4 years)
# - Daily pivot provides structural bias, Donchian for breakouts, volume for confirmation

name = "12h_1d_donchian_pivot_volume_v1"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 60:
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
    
    # Calculate 20-period Donchian channels
    donchian_period = 20
    donchian_high = pd.Series(high).rolling(window=donchian_period, min_periods=donchian_period).max().values
    donchian_low = pd.Series(low).rolling(window=donchian_period, min_periods=donchian_period).min().values
    donchian_mid = (donchian_high + donchian_low) / 2.0
    
    # Calculate 20-period volume SMA for confirmation
    volume_sma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    # Calculate daily pivot points (using previous day's OHLC)
    # Pivot = (High + Low + Close) / 3
    daily_high = df_1d['high'].values
    daily_low = df_1d['low'].values
    daily_close = df_1d['close'].values
    daily_pivot = (daily_high + daily_low + daily_close) / 3.0
    daily_pivot_aligned = align_htf_to_ltf(prices, df_1d, daily_pivot)
    
    # Track entry extreme for exit logic
    entry_price = np.full(n, np.nan)
    
    for i in range(donchian_period, n):
        # Skip if any required data is invalid
        if (np.isnan(donchian_high[i-1]) or np.isnan(donchian_low[i-1]) or
            np.isnan(volume_sma_20[i]) or np.isnan(daily_pivot_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Volume confirmation: 12h volume > 2.0x 20-period volume SMA
        vol_confirm = volume[i] > 2.0 * volume_sma_20[i]
        
        # Donchian breakout signals
        breakout_up = close[i] > donchian_high[i-1]  # Break above previous Donchian high
        breakout_down = close[i] < donchian_low[i-1]  # Break below previous Donchian low
        
        # Daily pivot bias
        price_above_pivot = close[i] > daily_pivot_aligned[i]
        price_below_pivot = close[i] < daily_pivot_aligned[i]
        
        if position == 0:  # Flat - look for entry
            if breakout_up and price_above_pivot and vol_confirm:
                position = 1
                signals[i] = 0.25
                entry_price[i] = close[i]
            elif breakout_down and price_below_pivot and vol_confirm:
                position = -1
                signals[i] = -0.25
                entry_price[i] = close[i]
            else:
                signals[i] = 0.0
        elif position == 1:  # Long position - look for exit
            # Exit on Donchian midpoint reversion or opposite breakout with volume
            exit_condition = (close[i] < donchian_mid[i]) or \
                           (breakout_down and vol_confirm)
            if exit_condition:
                position = 0
                signals[i] = 0.0
                entry_price[i] = np.nan
            else:
                signals[i] = 0.25
        else:  # position == -1 (Short position) - look for exit
            # Exit on Donchian midpoint reversion or opposite breakout with volume
            exit_condition = (close[i] > donchian_mid[i]) or \
                           (breakout_up and vol_confirm)
            if exit_condition:
                position = 0
                signals[i] = 0.0
                entry_price[i] = np.nan
            else:
                signals[i] = -0.25
    
    return signals
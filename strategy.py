#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1d Donchian(20) breakout with weekly EMA(21) trend filter and volume confirmation
# - Long when price breaks above 20-period Donchian high AND price > weekly EMA(21)
# - Short when price breaks below 20-period Donchian low AND price < weekly EMA(21)
# - Volume confirmation: 1d volume > 1.5x 20-period 1d volume SMA
# - Exit: Donchian midpoint reversion
# - Position sizing: 0.25 discrete level
# - Target: 7-25 trades/year on 1d timeframe (30-100 total over 4 years)
# - Weekly EMA provides structural bias for BTC/ETH in both bull and bear markets
# - Donchian breakout captures momentum, volume confirmation reduces false signals

name = "1d_1w_donchian_ema_volume_v1"
timeframe = "1d"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 60:
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
    
    # Calculate 20-period Donchian channels
    donchian_period = 20
    donchian_high = pd.Series(high).rolling(window=donchian_period, min_periods=donchian_period).max().values
    donchian_low = pd.Series(low).rolling(window=donchian_period, min_periods=donchian_period).min().values
    donchian_mid = (donchian_high + donchian_low) / 2.0
    
    # Calculate 20-period volume SMA for confirmation
    volume_sma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    # Calculate weekly EMA(21) for trend filter
    weekly_close = df_1w['close'].values
    weekly_ema_21 = pd.Series(weekly_close).ewm(span=21, adjust=False, min_periods=21).mean().values
    weekly_ema_21_aligned = align_htf_to_ltf(prices, df_1w, weekly_ema_21)
    
    for i in range(donchian_period, n):
        # Skip if any required data is invalid
        if (np.isnan(donchian_high[i-1]) or np.isnan(donchian_low[i-1]) or
            np.isnan(volume_sma_20[i]) or np.isnan(weekly_ema_21_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Volume confirmation: 1d volume > 1.5x 20-period volume SMA
        vol_confirm = volume[i] > 1.5 * volume_sma_20[i]
        
        # Donchian breakout signals
        breakout_up = close[i] > donchian_high[i-1]  # Break above previous Donchian high
        breakout_down = close[i] < donchian_low[i-1]  # Break below previous Donchian low
        
        # Weekly EMA trend filter
        price_above_ema = close[i] > weekly_ema_21_aligned[i]
        price_below_ema = close[i] < weekly_ema_21_aligned[i]
        
        if position == 0:  # Flat - look for entry
            if breakout_up and price_above_ema and vol_confirm:
                position = 1
                signals[i] = 0.25
            elif breakout_down and price_below_ema and vol_confirm:
                position = -1
                signals[i] = -0.25
            else:
                signals[i] = 0.0
        elif position == 1:  # Long position - look for exit
            # Exit on Donchian midpoint reversion
            if close[i] < donchian_mid[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
        else:  # position == -1 (Short position) - look for exit
            # Exit on Donchian midpoint reversion
            if close[i] > donchian_mid[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
    
    return signals
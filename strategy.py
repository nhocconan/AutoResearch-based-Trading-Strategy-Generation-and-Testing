#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Donchian(20) breakout with 1d trend filter (price above/below 200 EMA) and volume confirmation.
# Long when: price breaks above Donchian upper (20-period high) + price > 1d EMA200 + volume > 1.5x 20-period avg volume
# Short when: price breaks below Donchian lower (20-period low) + price < 1d EMA200 + volume > 1.5x 20-period avg volume
# Exit when price crosses back through the Donchian midpoint or trend reverses.
# Uses price breakouts for momentum, higher timeframe for trend alignment, and volume to avoid false breakouts.
# Designed for ~20-30 trades/year per symbol.
name = "4h_Donchian20_1dEMA200_VolumeConfirm"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # 1d data for EMA trend filter
    df_1d = get_htf_data(prices, '1d')
    close_1d = df_1d['close'].values
    ema_1d_200 = pd.Series(close_1d).ewm(span=200, adjust=False, min_periods=200).mean().values
    ema_1d_200_aligned = align_htf_to_ltf(prices, df_1d, ema_1d_200)
    
    # Donchian channels (20-period)
    donchian_upper = pd.Series(high).rolling(window=20, min_periods=20).max().values
    donchian_lower = pd.Series(low).rolling(window=20, min_periods=20).min().values
    donchian_middle = (donchian_upper + donchian_lower) / 2
    
    # Volume confirmation: current volume > 1.5x 20-period average volume
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_confirm = volume > (1.5 * vol_ma_20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 20  # Wait for Donchian calculation
    
    for i in range(start_idx, n):
        # Skip if any required data is not available
        if (np.isnan(ema_1d_200_aligned[i]) or np.isnan(donchian_upper[i]) or 
            np.isnan(donchian_lower[i]) or np.isnan(donchian_middle[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        upper = donchian_upper[i]
        lower = donchian_lower[i]
        middle = donchian_middle[i]
        ema_1d = ema_1d_200_aligned[i]
        vol_confirm = volume_confirm[i]
        
        if position == 0:
            # Long: break above upper + price above 1d EMA200 + volume confirmation
            if price > upper and price > ema_1d and vol_confirm:
                signals[i] = 0.25
                position = 1
            # Short: break below lower + price below 1d EMA200 + volume confirmation
            elif price < lower and price < ema_1d and vol_confirm:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Long exit: price crosses below middle OR trend reverses (price < 1d EMA200)
            if price < middle or price < ema_1d:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: price crosses above middle OR trend reverses (price > 1d EMA200)
            if price > middle or price > ema_1d:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals
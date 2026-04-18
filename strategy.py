#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1d Donchian breakout with 1w trend filter and volume confirmation.
# Long when price breaks above Donchian(20) high, 1w EMA(21) is rising, and volume > 1.5x SMA(20)
# Short when price breaks below Donchian(20) low, 1w EMA(21) is falling, and volume > 1.5x SMA(20)
# Exit when price reverses to the opposite Donchian boundary or volume drops.
# Uses price channel breakout for entry, weekly trend for direction, and volume filter to avoid false breakouts.
# Designed for ~15-25 trades/year per symbol.
name = "1d_Donchian20_Volume_1wTrend"
timeframe = "1d"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # 1d Donchian channels
    donchian_high = pd.Series(high).rolling(window=20, min_periods=20).max().values
    donchian_low = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # 1d volume filter
    vol_sma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    # 1w data for EMA trend filter
    df_1w = get_htf_data(prices, '1w')
    close_1w = df_1w['close'].values
    ema_1w_21 = pd.Series(close_1w).ewm(span=21, adjust=False, min_periods=21).mean().values
    ema_1w_21_prev = np.roll(ema_1w_21, 1)
    ema_1w_21_prev[0] = ema_1w_21[0]
    ema_1w_rising = ema_1w_21 > ema_1w_21_prev
    ema_1w_falling = ema_1w_21 < ema_1w_21_prev
    ema_1w_rising_aligned = align_htf_to_ltf(prices, df_1w, ema_1w_rising)
    ema_1w_falling_aligned = align_htf_to_ltf(prices, df_1w, ema_1w_falling)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 40  # Wait for indicator calculations
    
    for i in range(start_idx, n):
        # Skip if any required data is not available
        if (np.isnan(donchian_high[i]) or np.isnan(donchian_low[i]) or 
            np.isnan(vol_sma[i]) or np.isnan(ema_1w_rising_aligned[i]) or 
            np.isnan(ema_1w_falling_aligned[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        vol = volume[i]
        vol_sma_val = vol_sma[i]
        ema_rising = ema_1w_rising_aligned[i]
        ema_falling = ema_1w_falling_aligned[i]
        
        if position == 0:
            # Long: break above Donchian high, weekly EMA rising, volume > 1.5x SMA
            if price > donchian_high[i] and ema_rising and vol > 1.5 * vol_sma_val:
                signals[i] = 0.25
                position = 1
            # Short: break below Donchian low, weekly EMA falling, volume > 1.5x SMA
            elif price < donchian_low[i] and ema_falling and vol > 1.5 * vol_sma_val:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Long exit: price breaks below Donchian low or volume drops below SMA
            if price < donchian_low[i] or vol < vol_sma_val:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: price breaks above Donchian high or volume drops below SMA
            if price > donchian_high[i] or vol < vol_sma_val:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals
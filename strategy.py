#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Donchian(20) breakout + 1d HMA trend filter + volume spike + ATR stoploss
# - Long when price breaks above 4h Donchian upper band + 1d HMA(21) rising + 1d volume > 2.0x 20-period average
# - Short when price breaks below 4h Donchian lower band + 1d HMA(21) falling + 1d volume > 2.0x 20-period average
# - Exit when price reverts to 4h Donchian middle band or ATR stoploss triggered (adverse move > 2.5*ATR)
# - Uses discrete position sizing (0.25) to minimize fee churn
# - HMA filter ensures we only trade in the direction of the daily trend, reducing false breakouts
# - Volume spike confirms institutional participation
# - Target: 20-40 trades/year to stay within fee drag limits while capturing strong moves

name = "4h_1d_donchian_hma_volume_v1"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    entry_price = 0.0
    
    # Load 4h data ONCE before loop for Donchian bands and ATR (MTF rule compliance)
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 50:
        return signals
    
    # Load 1d data ONCE before loop for HMA and volume (MTF rule compliance)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return signals
    
    # Pre-compute 4h Donchian bands (20-period)
    high_4h = df_4h['high'].values
    low_4h = df_4h['low'].values
    close_4h = df_4h['close'].values
    
    # Donchian upper band: 20-period high
    donchian_upper = pd.Series(high_4h).rolling(window=20, min_periods=20).max().values
    # Donchian lower band: 20-period low
    donchian_lower = pd.Series(low_4h).rolling(window=20, min_periods=20).min().values
    # Donchian middle band: midpoint
    donchian_middle = (donchian_upper + donchian_lower) / 2.0
    
    # Align Donchian bands to 4h timeframe
    donchian_upper_aligned = align_htf_to_ltf(prices, df_4h, donchian_upper)
    donchian_lower_aligned = align_htf_to_ltf(prices, df_4h, donchian_lower)
    donchian_middle_aligned = align_htf_to_ltf(prices, df_4h, donchian_middle)
    
    # Pre-compute 4h ATR(20) for stoploss
    tr1 = pd.Series(high_4h).rolling(2).max() - pd.Series(low_4h).rolling(2).min()
    tr2 = abs(pd.Series(high_4h).shift(1) - pd.Series(close_4h))
    tr3 = abs(pd.Series(low_4h).shift(1) - pd.Series(close_4h))
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr_20 = tr.rolling(window=20, min_periods=20).mean().values
    atr_20_aligned = align_htf_to_ltf(prices, df_4h, atr_20)
    
    # Pre-compute 1d HMA(21) for trend filter
    # HMA = WMA(2 * WMA(n/2) - WMA(n)), sqrt(n)
    half_len = 21 // 2
    sqrt_len = int(np.sqrt(21))
    
    def wma(values, window):
        weights = np.arange(1, window + 1)
        return np.convolve(values, weights / weights.sum(), mode='valid')
    
    close_1d = df_1d['close'].values
    if len(close_1d) >= 21:
        wma_half = wma(close_1d, half_len)
        wma_full = wma(close_1d, 21)
        # Handle array lengths
        raw_hma = 2 * wma_half[-len(wma_full):] - wma_full
        hma_21 = wma(raw_hma, sqrt_len)
        # Pad to match original length
        hma_21_padded = np.full(len(close_1d), np.nan)
        hma_21_padded[half_len - 1 + len(hma_21):] = hma_21
        hma_21_values = hma_21_padded
    else:
        hma_21_values = np.full(len(close_1d), np.nan)
    
    # HMA rising/falling
    hma_rising = np.zeros_like(hma_21_values, dtype=bool)
    hma_falling = np.zeros_like(hma_21_values, dtype=bool)
    hma_rising[1:] = hma_21_values[1:] > hma_21_values[:-1]
    hma_falling[1:] = hma_21_values[1:] < hma_21_values[:-1]
    
    # Align HMA trend to 4h timeframe
    hma_rising_aligned = align_htf_to_ltf(prices, df_1d, hma_rising)
    hma_falling_aligned = align_htf_to_ltf(prices, df_1d, hma_falling)
    
    # Pre-compute 1d volume SMA for confirmation
    volume_1d = df_1d['volume'].values
    volume_sma_20_1d = pd.Series(volume_1d).rolling(window=20, min_periods=20).mean().values
    volume_sma_20_1d_aligned = align_htf_to_ltf(prices, df_1d, volume_sma_20_1d)
    
    for i in range(100, n):  # Start after 100-bar warmup
        # Skip if any required data is invalid
        if (np.isnan(donchian_upper_aligned[i]) or np.isnan(donchian_lower_aligned[i]) or
            np.isnan(donchian_middle_aligned[i]) or np.isnan(atr_20_aligned[i]) or
            np.isnan(volume_sma_20_1d_aligned[i]) or np.isnan(hma_rising_aligned[i]) or
            np.isnan(hma_falling_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Current price data
        close_price = close[i]
        high_price = high[i]
        low_price = low[i]
        
        # Volume confirmation: 1d volume > 2.0x 20-period volume average (tight threshold)
        volume_1d_current = df_1d['volume'].values
        volume_1d_aligned = align_htf_to_ltf(prices, df_1d, volume_1d_current)
        vol_confirm = volume_1d_aligned[i] > 2.0 * volume_sma_20_1d_aligned[i]
        
        # HMA trend filter
        hma_up = hma_rising_aligned[i]
        hma_down = hma_falling_aligned[i]
        
        # Donchian breakout conditions
        breakout_up = close_price > donchian_upper_aligned[i]
        breakout_down = close_price < donchian_lower_aligned[i]
        
        # Entry conditions
        enter_long = breakout_up and hma_up and vol_confirm
        enter_short = breakout_down and hma_down and vol_confirm
        
        # Exit conditions
        exit_long = (position == 1 and 
                    (close_price < donchian_middle_aligned[i] or  # Revert to middle band
                     close_price < entry_price - 2.5 * atr_20_aligned[i]))  # ATR stoploss
        exit_short = (position == -1 and 
                     (close_price > donchian_middle_aligned[i] or  # Revert to middle band
                      close_price > entry_price + 2.5 * atr_20_aligned[i]))  # ATR stoploss
        
        # Trading logic
        if enter_long and position != 1:
            position = 1
            entry_price = close_price
            signals[i] = 0.25
        elif enter_short and position != -1:
            position = -1
            entry_price = close_price
            signals[i] = -0.25
        elif position == 1 and exit_long:
            position = 0
            entry_price = 0.0
            signals[i] = 0.0
        elif position == -1 and exit_short:
            position = 0
            entry_price = 0.0
            signals[i] = 0.0
        else:
            # Maintain current position
            signals[i] = 0.25 if position == 1 else (-0.25 if position == -1 else 0.0)
    
    return signals
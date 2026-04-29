#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Donchian(20) breakout with 1d EMA34 trend filter and volume confirmation
# Long when price breaks above Donchian upper (20) AND price > 1d EMA34 AND volume > 1.8x 20-bar avg
# Short when price breaks below Donchian lower (20) AND price < 1d EMA34 AND volume > 1.8x 20-bar avg
# Exit when price crosses opposite Donchian level (lower for longs, upper for shorts)
# Uses discrete position sizing (0.25) to minimize fee churn while capturing moves.
# Target: 75-200 total trades over 4 years (19-50/year) on 4h.
# Donchian channels provide objective breakout levels; 1d EMA34 filters counter-trend moves.
# Volume spike ensures institutional participation, reducing false breakouts.
# Works in bull markets (trend continuation via breakouts) and bear markets (mean reversion within trend via exits).

name = "4h_Donchian20_VolumeConfirm_1dEMA34_v1"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Get 1d data for EMA34 trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 34:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    # Calculate EMA(34) on 1d data
    ema_34_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    # Align EMA34 to 4h timeframe
    ema_34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # Get 1d data for Donchian channel calculation (using prior day's OHLC)
    if len(df_1d) < 1:
        return np.zeros(n)
    
    # Extract prior day's OHLC (1d timeframe)
    # We need the completed prior day's OHLC to calculate today's Donchian levels
    # Shift by 1 to use only completed prior day
    prior_high = np.roll(df_1d['high'].values, 1)
    prior_low = np.roll(df_1d['low'].values, 1)
    prior_close = np.roll(df_1d['close'].values, 1)
    # Set first value to NaN as we don't have prior day
    prior_high[0] = np.nan
    prior_low[0] = np.nan
    prior_close[0] = np.nan
    
    # Align prior day OHLC to 4h timeframe
    prior_high_aligned = align_htf_to_ltf(prices, df_1d, prior_high)
    prior_low_aligned = align_htf_to_ltf(prices, df_1d, prior_low)
    prior_close_aligned = align_htf_to_ltf(prices, df_1d, prior_close)
    
    # Calculate Donchian(20) levels for each 4h bar based on prior day's OHLC
    # Donchian Upper(20) = max(high, lookback=20) on prior day
    # Donchian Lower(20) = min(low, lookback=20) on prior day
    # We'll use a rolling window on the aligned prior day data
    # But since we only have daily data aligned, we need to compute Donchian on 1d then align
    
    # Calculate Donchian on 1d data first
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    
    # Donchian Upper(20) = rolling max of high over 20 periods
    donchian_high_1d = pd.Series(high_1d).rolling(window=20, min_periods=20).max().values
    # Donchian Lower(20) = rolling min of low over 20 periods
    donchian_low_1d = pd.Series(low_1d).rolling(window=20, min_periods=20).min().values
    
    # Align Donchian levels to 4h timeframe
    donchian_high_aligned = align_htf_to_ltf(prices, df_1d, donchian_high_1d)
    donchian_low_aligned = align_htf_to_ltf(prices, df_1d, donchian_low_1d)
    
    # Volume confirmation: >1.8x 20-bar average volume
    volume_series = pd.Series(volume)
    volume_ma_20 = volume_series.rolling(window=20, min_periods=20).mean().values
    volume_confirm = volume > 1.8 * volume_ma_20
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(34, 20) + 1  # EMA34 warmup, Donchian warmup, +1 for prior day shift
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(ema_34_1d_aligned[i]) or np.isnan(donchian_high_aligned[i]) or np.isnan(donchian_low_aligned[i]) or 
            np.isnan(volume_ma_20[i])):
            signals[i] = 0.0
            continue
        
        vol_conf = volume_confirm[i]
        curr_close = close[i]
        ema_34 = ema_34_1d_aligned[i]
        
        # Donchian levels
        donchian_high = donchian_high_aligned[i]
        donchian_low = donchian_low_aligned[i]
        
        # Handle exits and position management
        if position == 1:  # Long position
            # Exit: price crosses below Donchian lower (breakdown)
            if curr_close < donchian_low:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit: price crosses above Donchian higher (breakout)
            if curr_close > donchian_high:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
                
        else:  # Flat - look for new entries
            # Long when price breaks above Donchian higher AND price > 1d EMA34 AND volume confirmation
            if curr_close > donchian_high and curr_close > ema_34 and vol_conf:
                signals[i] = 0.25
                position = 1
            # Short when price breaks below Donchian lower AND price < 1d EMA34 AND volume confirmation
            elif curr_close < donchian_low and curr_close < ema_34 and vol_conf:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
    
    return signals
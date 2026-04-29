#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1d Donchian(20) breakout with 1w EMA34 trend filter and volume confirmation
# Long when price breaks above upper Donchian band AND price > 1w EMA34 AND volume > 2.0x 20-bar avg
# Short when price breaks below lower Donchian band AND price < 1w EMA34 AND volume > 2.0x 20-bar avg
# Exit when price crosses opposite Donchian band (lower for longs, upper for shorts)
# Uses discrete position sizing (0.25) to minimize fee churn while capturing moves.
# Target: 30-100 total trades over 4 years (7-25/year) on 1d.
# Donchian(20) provides robust price channels that work in both trending and ranging markets.
# 1w EMA34 filters counter-trend moves, volume spike ensures institutional participation.
# Works in bull markets (trend continuation via breakouts) and bear markets (mean reversion within trend via exits).

name = "1d_Donchian20_1wEMA34_VolumeConfirm_v1"
timeframe = "1d"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Get 1w data for EMA34 trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 34:
        return np.zeros(n)
    
    close_1w = df_1w['close'].values
    # Calculate EMA(34) on 1w data
    ema_34_1w = pd.Series(close_1w).ewm(span=34, adjust=False, min_periods=34).mean().values
    # Align EMA34 to 1d timeframe
    ema_34_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_34_1w)
    
    # Get 1d data for Donchian channels (using prior day's OHLC)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    
    # Extract prior day's OHLC (1d timeframe) for Donchian calculation
    # We need the completed prior 20 days' OHLC to calculate today's Donchian levels
    # Shift by 1 to use only completed prior days
    prior_high = np.roll(df_1d['high'].values, 1)
    prior_low = np.roll(df_1d['low'].values, 1)
    prior_close = np.roll(df_1d['close'].values, 1)
    # Set first value to NaN as we don't have prior day
    prior_high[0] = np.nan
    prior_low[0] = np.nan
    prior_close[0] = np.nan
    
    # Align prior day OHLC to 1d timeframe (already 1d, but need alignment for rolling)
    prior_high_aligned = align_htf_to_ltf(prices, df_1d, prior_high)
    prior_low_aligned = align_htf_to_ltf(prices, df_1d, prior_low)
    prior_close_aligned = align_htf_to_ltf(prices, df_1d, prior_close)
    
    # Calculate Donchian(20) channels for each 1d bar based on prior 20 days' OHLC
    # Upper band = highest high of prior 20 days
    # Lower band = lowest low of prior 20 days
    high_series = pd.Series(prior_high_aligned)
    low_series = pd.Series(prior_low_aligned)
    upper_band = high_series.rolling(window=20, min_periods=20).max().values
    lower_band = low_series.rolling(window=20, min_periods=20).min().values
    
    # Volume confirmation: >2.0x 20-bar average volume
    volume_series = pd.Series(volume)
    volume_ma_20 = volume_series.rolling(window=20, min_periods=20).mean().values
    volume_confirm = volume > 2.0 * volume_ma_20
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(34, 20) + 1  # EMA34 warmup + Donchian warmup + 1 for prior day shift
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(ema_34_1w_aligned[i]) or np.isnan(upper_band[i]) or np.isnan(lower_band[i]) or 
            np.isnan(volume_ma_20[i])):
            signals[i] = 0.0
            continue
        
        vol_conf = volume_confirm[i]
        curr_close = close[i]
        ema_34 = ema_34_1w_aligned[i]
        
        # Donchian levels
        upper_level = upper_band[i]
        lower_level = lower_band[i]
        
        # Handle exits and position management
        if position == 1:  # Long position
            # Exit: price crosses below lower band (mean reversion to median)
            if curr_close < lower_level:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit: price crosses above upper band (mean reversion to median)
            if curr_close > upper_level:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
                
        else:  # Flat - look for new entries
            # Long when price breaks above upper band AND price > 1w EMA34 AND volume confirmation
            if curr_close > upper_level and curr_close > ema_34 and vol_conf:
                signals[i] = 0.25
                position = 1
            # Short when price breaks below lower band AND price < 1w EMA34 AND volume confirmation
            elif curr_close < lower_level and curr_close < ema_34 and vol_conf:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
    
    return signals
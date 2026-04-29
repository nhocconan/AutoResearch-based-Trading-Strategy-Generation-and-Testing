#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h Donchian(20) breakout with 1d EMA34 trend filter and volume spike confirmation
# Long when price breaks above Donchian upper (20) AND price > 1d EMA34 AND volume > 2.0x 24-bar avg (12h bars = 12 days)
# Short when price breaks below Donchian lower (20) AND price < 1d EMA34 AND volume > 2.0x 24-bar avg
# Exit when price retouches Donchian midpoint or opposite breakout occurs
# Uses discrete position sizing (0.25) to minimize fee drag. Target: 12-37 trades/year on 12h.
# Donchian channels provide proven breakout structure with edge in trending markets.
# 1d EMA34 filter ensures we only trade with the higher timeframe trend, improving win rate and reducing whipsaws.
# Volume confirmation ensures breakouts have conviction, reducing false signals in choppy markets.
# This strategy focuses on BTC and ETH as primary targets, avoiding SOL-only bias.

name = "12h_Donchian20_1dEMA34_VolumeSpike_v1"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
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
    # Align EMA34 to 12h timeframe
    ema_34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # Get prior day's OHLC for Donchian channels (using completed 1d bar)
    # We need to use completed daily bar, so we'll use 1d HTF data for OHLC with proper alignment
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Donchian channels: upper = max(high, lookback), lower = min(low, lookback)
    # Calculate from prior completed 1d bar with 20-day lookback
    high_series = pd.Series(high_1d)
    low_series = pd.Series(low_1d)
    donch_upper_1d = high_series.rolling(window=20, min_periods=20).max().values
    donch_lower_1d = low_series.rolling(window=20, min_periods=20).min().values
    donch_mid_1d = (donch_upper_1d + donch_lower_1d) / 2.0
    
    # Align Donchian levels to 12h timeframe (wait for daily bar to close)
    donch_upper_aligned = align_htf_to_ltf(prices, df_1d, donch_upper_1d)
    donch_lower_aligned = align_htf_to_ltf(prices, df_1d, donch_lower_1d)
    donch_mid_aligned = align_htf_to_ltf(prices, df_1d, donch_mid_1d)
    
    # Volume confirmation: >2.0x 24-bar average volume (24 * 12h = 12 days, reasonable lookback)
    volume_series = pd.Series(volume)
    volume_ma_24 = volume_series.rolling(window=24, min_periods=24).mean().values
    volume_confirm = volume > 2.0 * volume_ma_24
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 34  # EMA34 needs 34 bars
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(donch_upper_aligned[i]) or np.isnan(donch_lower_aligned[i]) or 
            np.isnan(donch_mid_aligned[i]) or np.isnan(ema_34_1d_aligned[i]) or 
            np.isnan(volume_ma_24[i])):
            signals[i] = 0.0
            continue
        
        vol_conf = volume_confirm[i]
        curr_close = close[i]
        curr_high = high[i]
        curr_low = low[i]
        donch_upper = donch_upper_aligned[i]
        donch_lower = donch_lower_aligned[i]
        donch_mid = donch_mid_aligned[i]
        ema_34 = ema_34_1d_aligned[i]
        
        # Handle entries and exits
        if position == 0:  # Flat - look for new entries
            # Long when price breaks above Donchian upper AND price > 1d EMA34 AND volume confirmation
            if curr_high > donch_upper and curr_close > ema_34 and vol_conf:
                signals[i] = 0.25
                position = 1
            # Short when price breaks below Donchian lower AND price < 1d EMA34 AND volume confirmation
            elif curr_low < donch_lower and curr_close < ema_34 and vol_conf:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:  # Long - exit when price retouches midpoint or breaks below lower
            if curr_close <= donch_mid or curr_low < donch_lower:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:  # Short - exit when price retouches midpoint or breaks above upper
            if curr_close >= donch_mid or curr_high > donch_upper:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals
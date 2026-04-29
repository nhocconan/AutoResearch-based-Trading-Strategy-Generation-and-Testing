#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h Donchian(20) breakout with 1d EMA50 trend filter and volume spike
# Long when price breaks above Donchian upper band AND price > 1d EMA50 AND volume > 2.0x 20-bar avg
# Short when price breaks below Donchian lower band AND price < 1d EMA50 AND volume > 2.0x 20-bar avg
# Exit when price retouches Donchian midpoint (mean reversion) or opposite breakout occurs
# Uses discrete position sizing (0.25) to minimize fee drag. Target: 12-37 trades/year on 12h.
# Donchian(20) provides clear structure with proven edge on SOLUSDT test Sharpe 1.10-1.38
# 1d EMA50 filter ensures we only trade with the longer-term trend, improving win rate in bear markets
# Volume confirmation ensures breakouts have conviction, reducing false signals in choppy markets
# 12h timeframe targets 50-150 total trades over 4 years to avoid fee drag

name = "12h_Donchian20_1dEMA50_VolumeSpike_v1"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 60:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Get 1d data for Donchian calculation and EMA50 trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:  # Need sufficient data for Donchian(20) and EMA50
        return np.zeros(n)
    
    # Calculate Donchian channels from prior 1d data (using prior day's data to avoid look-ahead)
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Prior day's OHLC (shifted by 1 to avoid look-ahead)
    prior_high = np.roll(high_1d, 1)
    prior_low = np.roll(low_1d, 1)
    prior_close = np.roll(close_1d, 1)
    # Set first value to NaN since we don't have prior day for the first bar
    prior_high[0] = np.nan
    prior_low[0] = np.nan
    prior_close[0] = np.nan
    
    # Donchian(20) calculations using prior 1d data
    # We need 20 prior days, so we'll calculate on the 1d timeframe then align
    high_series = pd.Series(prior_high)
    low_series = pd.Series(prior_low)
    
    # Donchian upper = max(high, 20) on prior days
    donchian_high = high_series.rolling(window=20, min_periods=20).max().values
    # Donchian lower = min(low, 20) on prior days
    donchian_low = low_series.rolling(window=20, min_periods=20).min().values
    # Donchian midpoint = (upper + lower) / 2
    donchian_mid = (donchian_high + donchian_low) / 2.0
    
    # Align Donchian levels to 12h timeframe
    donchian_high_aligned = align_htf_to_ltf(prices, df_1d, donchian_high)
    donchian_low_aligned = align_htf_to_ltf(prices, df_1d, donchian_low)
    donchian_mid_aligned = align_htf_to_ltf(prices, df_1d, donchian_mid)
    
    # Calculate EMA(50) on 1d close data for trend filter
    ema_50_1d = pd.Series(prior_close).ewm(span=50, adjust=False, min_periods=50).mean().values
    # Align EMA50 to 12h timeframe
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # Volume confirmation: >2.0x 20-bar average volume
    volume_series = pd.Series(volume)
    volume_ma_20 = volume_series.rolling(window=20, min_periods=20).mean().values
    volume_confirm = volume > 2.0 * volume_ma_20
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(20, 20)  # Donchian(20) and Volume MA(20) need 20 bars
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(donchian_high_aligned[i]) or np.isnan(donchian_low_aligned[i]) or 
            np.isnan(donchian_mid_aligned[i]) or np.isnan(ema_50_1d_aligned[i]) or 
            np.isnan(volume_ma_20[i])):
            signals[i] = 0.0
            continue
        
        vol_conf = volume_confirm[i]
        curr_close = close[i]
        curr_high = high[i]
        curr_low = low[i]
        upper = donchian_high_aligned[i]
        lower = donchian_low_aligned[i]
        midpoint = donchian_mid_aligned[i]
        ema_50 = ema_50_1d_aligned[i]
        
        # Handle entries and exits
        if position == 0:  # Flat - look for new entries
            # Long when price breaks above Donchian upper band AND price > 1d EMA50 AND volume confirmation
            if curr_high > upper and curr_close > ema_50 and vol_conf:
                signals[i] = 0.25
                position = 1
            # Short when price breaks below Donchian lower band AND price < 1d EMA50 AND volume confirmation
            elif curr_low < lower and curr_close < ema_50 and vol_conf:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:  # Long - exit when price retouches Donchian midpoint or breaks below Donchian lower band
            if curr_close <= midpoint or curr_low < lower:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:  # Short - exit when price retouches Donchian midpoint or breaks above Donchian upper band
            if curr_close >= midpoint or curr_high > upper:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals
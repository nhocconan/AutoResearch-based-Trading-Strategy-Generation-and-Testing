#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1d Donchian(20) breakout with 1w EMA50 trend filter and volume confirmation.
# Uses weekly EMA50 for long-term trend direction, breaks above/below daily Donchian channels for entry,
# confirmed by volume spike (>2.0x 20-bar MA). Designed for 1d timeframe to achieve 30-100 total trades
# over 4 years (7-25/year) with discrete sizing (0.25). Works in both bull and bear markets via
# volatility-based breakouts and tight entry conditions requiring confluence of structure, trend, and volume.

name = "1d_Donchian20_Breakout_1wEMA50_Trend_VolumeConfirm_v1"
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
    
    # Weekly HTF data for EMA50 trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 50:
        return np.zeros(n)
    
    # Weekly EMA50 trend filter
    ema_50_1w = pd.Series(df_1w['close']).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1d = align_htf_to_ltf(prices, df_1w, ema_50_1w)
    
    # Daily Donchian channels (20-period)
    high_rolling_max = pd.Series(high).rolling(window=20, min_periods=20).max().values
    low_rolling_min = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # Volume confirmation: current volume > 2.0 * 20-period average volume
    volume_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_confirm = volume > (volume_ma_20 * 2.0)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    # Start after warmup for all indicators
    start_idx = 50  # Need 50 for weekly EMA + 20 for Donchian
    
    for i in range(start_idx, n):
        if np.isnan(ema_50_1d[i]) or np.isnan(high_rolling_max[i]) or np.isnan(low_rolling_min[i]) or np.isnan(volume_ma_20[i]):
            signals[i] = 0.0
            if position != 0:
                position = 0
            continue
        
        curr_close = close[i]
        curr_high = high[i]
        curr_low = low[i]
        curr_volume = volume[i]
        
        # Volume confirmation
        vol_confirm = volume_confirm[i]
        
        if position == 0:  # Flat - look for new entries
            # Long: Price breaks above Donchian upper band, above weekly EMA50, and volume confirmation
            if curr_high > high_rolling_max[i] and curr_close > ema_50_1d[i] and vol_confirm:
                signals[i] = 0.25
                position = 1
                entry_price = curr_close
            # Short: Price breaks below Donchian lower band, below weekly EMA50, and volume confirmation
            elif curr_low < low_rolling_min[i] and curr_close < ema_50_1d[i] and vol_confirm:
                signals[i] = -0.25
                position = -1
                entry_price = curr_close
            else:
                signals[i] = 0.0
        
        elif position == 1:  # Long position
            # Exit on close below Donchian lower band (failed breakout) or below weekly EMA50
            if curr_close < low_rolling_min[i] or curr_close < ema_50_1d[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:  # Short position
            # Exit on close above Donchian upper band (failed breakdown) or above weekly EMA50
            if curr_close > high_rolling_max[i] or curr_close > ema_50_1d[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals
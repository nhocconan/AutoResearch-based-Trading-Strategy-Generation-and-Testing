#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Donchian(20) breakout with 12h EMA50 trend filter and volume confirmation
# Uses 12h EMA50 > EMA100 as trend filter (strong trend) + Donchian breakout from prior 4h bar
# Volume spike (2.0x 20-period MA) confirms participation
# Works in bull/bear via trend filter - only trades in strong directional moves
# Designed for low frequency (75-200 trades over 4 years) to minimize fee drag on 4h timeframe

name = "4h_Donchian20_12hEMA50_Trend_VolumeSpike_v1"
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
    
    # 12h HTF data for EMA trend filter
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 100:
        return np.zeros(n)
    
    # 12h EMA50 and EMA100 calculation (trend filter)
    close_12h = df_12h['close'].values
    ema_50 = pd.Series(close_12h).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_100 = pd.Series(close_12h).ewm(span=100, adjust=False, min_periods=100).mean().values
    
    # Trend: EMA50 > EMA100 for uptrend, EMA50 < EMA100 for downtrend
    ema_50_aligned = align_htf_to_ltf(prices, df_12h, ema_50)
    ema_100_aligned = align_htf_to_ltf(prices, df_12h, ema_100)
    
    # Calculate Donchian channels from prior 4h bar (using prior bar's HL)
    prior_high = np.concatenate([[np.nan], high[:-1]])
    prior_low = np.concatenate([[np.nan], low[:-1]])
    
    # Donchian(20): highest high and lowest low of prior 20 periods
    def rolling_max(arr, window):
        result = np.full_like(arr, np.nan)
        for i in range(window-1, len(arr)):
            result[i] = np.max(arr[i-window+1:i+1])
        return result
    
    def rolling_min(arr, window):
        result = np.full_like(arr, np.nan)
        for i in range(window-1, len(arr)):
            result[i] = np.min(arr[i-window+1:i+1])
        return result
    
    donchian_high = rolling_max(prior_high, 20)
    donchian_low = rolling_min(prior_low, 20)
    
    # Volume confirmation: current volume > 2.0 * 20-period average volume
    volume_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (volume_ma_20 * 2.0)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after warmup for all indicators
    start_idx = max(100, 20)  # Need EMA100 and Donchian20
    
    for i in range(start_idx, n):
        if (np.isnan(ema_50_aligned[i]) or np.isnan(ema_100_aligned[i]) or 
            np.isnan(donchian_high[i]) or np.isnan(donchian_low[i]) or 
            np.isnan(volume_ma_20[i])):
            signals[i] = 0.0
            continue
        
        # Trend filter conditions
        uptrend = ema_50_aligned[i] > ema_100_aligned[i]
        downtrend = ema_50_aligned[i] < ema_100_aligned[i]
        
        # Breakout conditions
        breakout_long = close[i] > donchian_high[i]  # Price breaks above Donchian high
        breakout_short = close[i] < donchian_low[i]  # Price breaks below Donchian low
        
        # Volume confirmation
        vol_spike = volume_spike[i]
        
        if position == 0:  # Flat - look for new entries
            # Long: Breakout above Donchian high with volume spike and uptrend
            if breakout_long and vol_spike and uptrend:
                signals[i] = 0.25
                position = 1
            # Short: Breakout below Donchian low with volume spike and downtrend
            elif breakout_short and vol_spike and downtrend:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        
        elif position == 1:  # Long position
            # Exit on close below Donchian low or trend reversal
            if close[i] < donchian_low[i] or not uptrend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:  # Short position
            # Exit on close above Donchian high or trend reversal
            if close[i] > donchian_high[i] or not downtrend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals
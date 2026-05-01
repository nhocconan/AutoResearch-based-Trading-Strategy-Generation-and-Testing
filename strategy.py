#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h Donchian(20) breakout with 1d EMA34 trend filter and volume confirmation
# Uses daily EMA for major trend direction (filters counter-trend breakouts)
# Donchian(20) provides clear breakout levels from 12h price action
# Volume spike confirms breakout authenticity
# Designed for low frequency (50-150 trades over 4 years) to minimize fee drag
# Works in bull/bear via trend filter + breakout logic

name = "12h_Donchian20_1dEMA34_Trend_VolumeSpike_v1"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # 1d HTF data for EMA34 trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 34:
        return np.zeros(n)
    
    # 1d EMA34 for trend filter
    close_1d = df_1d['close'].values
    ema_34_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # Volume confirmation: current volume > 2.0 * 20-period average volume
    volume_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (volume_ma_20 * 2.0)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after warmup for all indicators
    start_idx = max(34, 20)  # Need 1d EMA34 and volume MA20
    
    for i in range(start_idx, n):
        if (np.isnan(ema_34_1d_aligned[i]) or np.isnan(volume_ma_20[i])):
            signals[i] = 0.0
            continue
        
        # Calculate Donchian(20) from 12h data (lookback 20 bars, exclude current)
        lookback_start = max(0, i - 20)
        lookback_end = i  # exclude current bar
        if lookback_end - lookback_start < 20:
            # Not enough lookback data, skip
            signals[i] = 0.0 if position == 0 else (0.25 if position == 1 else -0.25)
            continue
            
        highest_high = np.max(high[lookback_start:lookback_end])
        lowest_low = np.min(low[lookback_start:lookback_end])
        
        # Breakout conditions using Donchian levels
        breakout_up = close[i] > highest_high  # Break above upper band
        breakout_down = close[i] < lowest_low  # Break below lower band
        
        # Trend filter: price above/below 1d EMA34
        uptrend = close[i] > ema_34_1d_aligned[i]
        downtrend = close[i] < ema_34_1d_aligned[i]
        
        # Volume confirmation
        vol_spike = volume_spike[i]
        
        if position == 0:  # Flat - look for new entries
            # Long: upward breakout above upper band, volume spike, uptrend
            if breakout_up and vol_spike and uptrend:
                signals[i] = 0.25
                position = 1
            # Short: downward breakout below lower band, volume spike, downtrend
            elif breakout_down and vol_spike and downtrend:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        
        elif position == 1:  # Long position
            # Exit on trend reversal or price re-enters Donchian channel (below upper band)
            if not uptrend or close[i] < highest_high:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:  # Short position
            # Exit on trend reversal or price re-enters Donchian channel (above lower band)
            if not downtrend or close[i] > lowest_low:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals
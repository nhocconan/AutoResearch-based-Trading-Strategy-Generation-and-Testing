#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get weekly data for higher timeframe trend
    df_1w = get_htf_data(prices, '1w')
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    volume_1w = df_1w['volume'].values
    
    # Calculate weekly EMA34 for trend filter
    ema34_1w = pd.Series(close_1w).ewm(span=34, adjust=False, min_periods=34).mean().values
    
    # Calculate weekly volume spike (volume > 2.0x 12-period average)
    vol_ma_1w = pd.Series(volume_1w).rolling(window=12, min_periods=12).mean().values
    volume_spike_1w = volume_1w > (2.0 * vol_ma_1w)
    
    # Align weekly indicators to daily timeframe
    ema34_1w_aligned = align_htf_to_ltf(prices, df_1w, ema34_1w)
    volume_spike_1w_aligned = align_htf_to_ltf(prices, df_1w, volume_spike_1w.astype(float))
    
    # Calculate daily Donchian channels (20-period)
    donchian_high = pd.Series(high).rolling(window=20, min_periods=20).max().values
    donchian_low = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 60  # Wait for enough data
    
    for i in range(start_idx, n):
        # Skip if any required data is not available
        if (np.isnan(ema34_1w_aligned[i]) or
            np.isnan(donchian_high[i]) or
            np.isnan(donchian_low[i])):
            signals[i] = 0.0
            continue
        
        trade_allowed = volume_spike_1w_aligned[i]
        
        if position == 0:
            # Long: Donchian breakout above upper band with weekly EMA34 uptrend
            if trade_allowed and close[i] > donchian_high[i] and close[i] > ema34_1w_aligned[i]:
                signals[i] = 0.25
                position = 1
            # Short: Donchian breakdown below lower band with weekly EMA34 downtrend
            elif trade_allowed and close[i] < donchian_low[i] and close[i] < ema34_1w_aligned[i]:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Long exit: price closes below weekly EMA34 or Donchian lower band
            if close[i] < ema34_1w_aligned[i] or close[i] < donchian_low[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: price closes above weekly EMA34 or Donchian upper band
            if close[i] > ema34_1w_aligned[i] or close[i] > donchian_high[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "1d_WeeklyEMA34_VolumeSpike_Donchian20"
timeframe = "1d"
leverage = 1.0
#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h Donchian(20) breakout with 1d EMA34 trend filter and volume spike confirmation
# Donchian(20) provides clear structure for breakout entries in both bull and bear markets
# 1d EMA34 filters for higher-timeframe trend alignment to avoid counter-trend trades
# Volume spike (>1.8 x 30-period EMA) confirms breakout validity and reduces false signals
# Discrete position sizing (0.25) balances opportunity with fee drag control
# Target: 50-150 total trades over 4 years (12-37/year) for optimal risk-adjusted returns

name = "12h_Donchian20_Breakout_1dEMA34_Trend_VolumeSpike"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 60:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Volume confirmation (volume spike > 1.8 x 30-period EMA)
    vol_ema_30 = pd.Series(volume).ewm(span=30, adjust=False, min_periods=30).mean().values
    volume_confirmation = volume > (1.8 * vol_ema_30)
    
    # 1d data for Donchian channels and EMA34
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 34:
        return np.zeros(n)
    
    # Donchian(20) channels from previous 1d bar
    donchian_high = pd.Series(df_1d['high'].values).rolling(window=20, min_periods=20).max().shift(1).values
    donchian_low = pd.Series(df_1d['low'].values).rolling(window=20, min_periods=20).min().shift(1).values
    
    # Align Donchian levels to 12h timeframe (wait for completed 1d bar)
    donchian_high_aligned = align_htf_to_ltf(prices, df_1d, donchian_high)
    donchian_low_aligned = align_htf_to_ltf(prices, df_1d, donchian_low)
    
    # 1d EMA34 for trend filter
    ema_34_1d = pd.Series(df_1d['close'].values).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after warmup (need enough data for calculations)
    start_idx = 60
    
    for i in range(start_idx, n):
        if (np.isnan(ema_34_1d_aligned[i]) or np.isnan(donchian_high_aligned[i]) or 
            np.isnan(donchian_low_aligned[i]) or np.isnan(volume_confirmation[i])):
            signals[i] = 0.0
            continue
        
        # Determine trend bias from 1d EMA34
        uptrend = close[i] > ema_34_1d_aligned[i]
        downtrend = close[i] < ema_34_1d_aligned[i]
        
        if position == 0:  # Flat - look for new entries
            # Long: Close breaks above Donchian high with volume confirmation and uptrend
            if close[i] > donchian_high_aligned[i] and volume_confirmation[i] and uptrend:
                signals[i] = 0.25
                position = 1
            # Short: Close breaks below Donchian low with volume confirmation and downtrend
            elif close[i] < donchian_low_aligned[i] and volume_confirmation[i] and downtrend:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        
        elif position == 1:  # Long position
            # Exit: Close drops below Donchian low (reversal to downside) OR trend changes to downtrend
            if close[i] < donchian_low_aligned[i] or not uptrend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:  # Short position
            # Exit: Close rises above Donchian high (reversal to upside) OR trend changes to uptrend
            if close[i] > donchian_high_aligned[i] or not downtrend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals
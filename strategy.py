#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1d Donchian(20) breakout with 1w EMA34 trend filter and volume confirmation (>1.5x 20-bar MA)
# Long when price breaks above Donchian upper channel (20-bar high) and close > 1w EMA34 and volume spike
# Short when price breaks below Donchian lower channel (20-bar low) and close < 1w EMA34 and volume spike
# Uses 1w EMA34 for higher-timeframe trend alignment to reduce whipsaws in ranging markets.
# Volume spike confirms institutional participation. Discrete sizing (0.25) minimizes fee churn.
# Target: 30-100 total trades over 4 years (7-25/year) to stay within fee drag limits.

name = "1d_Donchian20_Breakout_1wEMA34_Trend_VolumeSpike_v1"
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
    
    # 1w HTF data for EMA calculation
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 34:
        return np.zeros(n)
    
    # 1w EMA(34) on 1w close
    ema_1w_34 = pd.Series(df_1w['close']).ewm(span=34, adjust=False, min_periods=34).mean().values
    
    # Align 1w EMA to 1d timeframe
    ema_1w_34_aligned = align_htf_to_ltf(prices, df_1w, ema_1w_34)
    
    # Donchian channels (20-period)
    high_roll = pd.Series(high).rolling(window=20, min_periods=20).max().values
    low_roll = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # Volume confirmation: current volume > 1.5 * 20-period average volume
    volume_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (volume_ma_20 * 1.5)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after warmup for all indicators
    start_idx = 20  # Need 20 for Donchian and volume MA
    
    for i in range(start_idx, n):
        if (np.isnan(ema_1w_34_aligned[i]) or np.isnan(high_roll[i]) or 
            np.isnan(low_roll[i]) or np.isnan(volume_ma_20[i])):
            signals[i] = 0.0
            continue
        
        curr_close = close[i]
        curr_high = high[i]
        curr_low = low[i]
        curr_volume = volume[i]
        
        # Volume confirmation
        vol_spike = volume_spike[i]
        
        # Donchian breakout conditions
        breakout_up = curr_close > high_roll[i]
        breakout_down = curr_close < low_roll[i]
        
        if position == 0:  # Flat - look for new entries
            # Long: Price breaks above upper Donchian, close > 1w EMA34, volume spike
            if breakout_up and curr_close > ema_1w_34_aligned[i] and vol_spike:
                signals[i] = 0.25
                position = 1
            # Short: Price breaks below lower Donchian, close < 1w EMA34, volume spike
            elif breakout_down and curr_close < ema_1w_34_aligned[i] and vol_spike:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        
        elif position == 1:  # Long position
            # Exit on close below lower Donchian (trailing stop via signal)
            if curr_close < low_roll[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:  # Short position
            # Exit on close above upper Donchian (trailing stop via signal)
            if curr_close > high_roll[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals
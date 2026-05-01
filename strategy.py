#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1d Donchian(20) breakout with weekly ATR-filtered momentum and volume confirmation.
# Long when price breaks above Donchian(20) high AND weekly ATR(14) > 20-bar mean ATR AND volume > 1.5x 20-bar average.
# Short when price breaks below Donchian(20) low AND weekly ATR(14) > 20-bar mean ATR AND volume > 1.5x 20-bar average.
# Uses discrete sizing 0.25 to manage drawdown. Target: 50-100 total trades over 4 years (12-25/year).
# Weekly ATR filter ensures breakouts occur in sufficient volatility regimes, reducing false signals in low-vol chop.
# Volume confirmation threshold set to 1.5x to balance signal quality and trade frequency.
# Primary timeframe: 1d, HTF: 1w for weekly momentum filter.

name = "1d_Donchian20_WeeklyATR_VolumeSpike_v1"
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
    
    # Load 1w data ONCE before loop for weekly ATR filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 14:
        return np.zeros(n)
    
    # Calculate weekly ATR(14)
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    tr1 = np.maximum(high_1w[1:] - low_1w[1:], np.abs(high_1w[1:] - close_1w[:-1]))
    tr2 = np.maximum(np.abs(low_1w[1:] - close_1w[:-1]), tr1)
    tr = np.concatenate([[np.nan], tr2])  # first TR is NaN
    atr_1w = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    # Calculate 20-bar mean of weekly ATR for regime filter
    atr_ma_20 = pd.Series(atr_1w).rolling(window=20, min_periods=20).mean().values
    atr_regime = atr_1w > atr_ma_20  # True when weekly ATR above its 20-bar mean
    
    # Align weekly ATR regime to daily timeframe
    atr_regime_aligned = align_htf_to_ltf(prices, df_1w, atr_regime.astype(float))
    
    # Calculate Donchian(20) channels from daily data
    donchian_high = pd.Series(high).rolling(window=20, min_periods=20).max().shift(1).values
    donchian_low = pd.Series(low).rolling(window=20, min_periods=20).min().shift(1).values
    
    # Volume confirmation: current daily volume > 1.5x 20-bar average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 40  # warmup for Donchian and indicators
    
    for i in range(start_idx, n):
        if np.isnan(donchian_high[i]) or np.isnan(donchian_low[i]) or \
           np.isnan(atr_regime_aligned[i]) or np.isnan(vol_ma[i]):
            signals[i] = 0.0
            continue
        
        curr_close = close[i]
        curr_high = high[i]
        curr_low = low[i]
        curr_vol = volume[i]
        curr_vol_ma = vol_ma[i]
        
        if curr_vol_ma <= 0:
            signals[i] = 0.0
            continue
            
        volume_confirm = curr_vol > (curr_vol_ma * 1.5)  # Volume spike threshold
        vol_regime = atr_regime_aligned[i] == 1.0  # Weekly ATR regime filter
        
        # Donchian breakout signals
        breakout_up = curr_high > donchian_high[i]  # break above Donchian high
        breakout_down = curr_low < donchian_low[i]  # break below Donchian low
        
        # Entry conditions
        if position == 0:  # Flat - look for new entries
            # Long: breakout above Donchian high AND weekly ATR regime AND volume confirmation
            if (breakout_up and 
                vol_regime and 
                volume_confirm):
                signals[i] = 0.25
                position = 1
            # Short: breakout below Donchian low AND weekly ATR regime AND volume confirmation
            elif (breakout_down and 
                  vol_regime and 
                  volume_confirm):
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        
        elif position == 1:  # Long position
            # Exit: price crosses below Donchian low (stoploss) OR weekly ATR regime turns off
            if (curr_low < donchian_low[i] or 
                atr_regime_aligned[i] == 0.0):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:  # Short position
            # Exit: price crosses above Donchian high (stoploss) OR weekly ATR regime turns off
            if (curr_high > donchian_high[i] or 
                atr_regime_aligned[i] == 0.0):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals
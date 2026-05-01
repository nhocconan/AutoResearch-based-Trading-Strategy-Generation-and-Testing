#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1d Donchian(20) breakout with weekly ATR-based volatility filter and volume confirmation.
# Long when price breaks above Donchian(20) high AND weekly ATR ratio > 1.2 (expanding volatility) AND volume > 1.5x 20-day average.
# Short when price breaks below Donchian(20) low AND weekly ATR ratio > 1.2 AND volume > 1.5x 20-day average.
# Uses discrete sizing 0.25 to manage drawdown. Target: 30-100 total trades over 4 years (7-25/year).
# Weekly ATR filter ensures we only trade during volatile, trending markets, reducing false breakouts in ranging conditions.
# Volume confirmation adds conviction to breakouts. Primary timeframe: 1d, HTF: 1w for volatility regime.

name = "1d_Donchian20_WeeklyATR_VolumeConfirm_v1"
timeframe = "1d"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 60:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Load 1w data ONCE before loop for weekly volatility regime
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 2:
        return np.zeros(n)
    
    # Weekly ATR(14) for volatility regime
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    
    tr1 = high_1w - low_1w
    tr2 = np.abs(high_1w - np.roll(close_1w, 1))
    tr3 = np.abs(low_1w - np.roll(close_1w, 1))
    tr1[0] = 0  # first bar has no previous close
    tr2[0] = 0
    tr3[0] = 0
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    atr_1w = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    # Weekly ATR ratio: current ATR / 20-period average ATR (expanding volatility filter)
    atr_ma_20w = pd.Series(atr_1w).rolling(window=20, min_periods=20).mean().values
    atr_ratio = np.where(atr_ma_20w > 0, atr_1w / atr_ma_20w, 1.0)
    atr_ratio_aligned = align_htf_to_ltf(prices, df_1w, atr_ratio)
    
    # Calculate Donchian(20) channels from 1d data
    if len(close) < 20:
        return np.zeros(n)
    
    # Donchian high: max(high, 20) from previous completed day
    donchian_high = pd.Series(high).rolling(window=20, min_periods=20).max().shift(1).values
    # Donchian low: min(low, 20) from previous completed day
    donchian_low = pd.Series(low).rolling(window=20, min_periods=20).min().shift(1).values
    
    # Volume confirmation: current 1d volume > 1.5x 20-day average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 50  # warmup for Donchian and indicators
    
    for i in range(start_idx, n):
        if np.isnan(donchian_high[i]) or np.isnan(donchian_low[i]) or \
           np.isnan(atr_ratio_aligned[i]) or np.isnan(vol_ma[i]):
            signals[i] = 0.0
            continue
        
        curr_close = close[i]
        curr_high = high[i]
        curr_low = low[i]
        curr_vol = volume[i]
        curr_vol_ma = vol_ma[i]
        curr_atr_ratio = atr_ratio_aligned[i]
        
        if curr_vol_ma <= 0:
            signals[i] = 0.0
            continue
            
        volume_confirm = curr_vol > (curr_vol_ma * 1.5)  # Volume confirmation threshold
        volatility_filter = curr_atr_ratio > 1.2  # Only trade in expanding volatility
        
        # Donchian breakout signals
        breakout_up = curr_high > donchian_high[i]  # break above Donchian high
        breakout_down = curr_low < donchian_low[i]  # break below Donchian low
        
        # Entry conditions
        if position == 0:  # Flat - look for new entries
            # Long: breakout above Donchian high AND volatility filter AND volume confirmation
            if (breakout_up and 
                volatility_filter and 
                volume_confirm):
                signals[i] = 0.25
                position = 1
            # Short: breakout below Donchian low AND volatility filter AND volume confirmation
            elif (breakout_down and 
                  volatility_filter and 
                  volume_confirm):
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        
        elif position == 1:  # Long position
            # Exit: price crosses below Donchian low (stoploss) OR volatility deteriorates
            if (curr_low < donchian_low[i] or 
                atr_ratio_aligned[i] < 1.0):  # volatility contracting
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:  # Short position
            # Exit: price crosses above Donchian high (stoploss) OR volatility deteriorates
            if (curr_high > donchian_high[i] or 
                atr_ratio_aligned[i] < 1.0):  # volatility contracting
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals
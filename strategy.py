#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Donchian(20) breakout with 1d/1w regime filter and volume confirmation.
# Uses 1d ADX > 25 for trend strength and 1w close > 1w SMA50 for bull regime.
# Long when price breaks above Donchian(20) high AND 1d ADX > 25 AND 1w close > 1w SMA50 AND volume > 2x 20-bar average.
# Short when price breaks below Donchian(20) low AND 1d ADX > 25 AND 1w close < 1w SMA50 AND volume > 2x 20-bar average.
# Uses discrete sizing 0.25 to manage drawdown. Target: 50-150 total trades over 4 years.
# Volume spike threshold set to 2.0x to reduce whipsaws and fee drag.

name = "6h_Donchian20_1dADX25_1wRegime_VolumeSpike_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Load 1d data ONCE before loop for ADX trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:  # Need enough for ADX calculation
        return np.zeros(n)
    
    # Load 1w data ONCE before loop for regime filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 50:  # Need enough for SMA50
        return np.zeros(n)
    
    # 1d ADX calculation (14-period)
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # True Range
    tr1 = high_1d[1:] - low_1d[1:]
    tr2 = np.abs(high_1d[1:] - close_1d[:-1])
    tr3 = np.abs(low_1d[1:] - close_1d[:-1])
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr = np.concatenate([[np.nan], tr])  # First value is NaN
    
    # Directional Movement
    up_move = high_1d[1:] - high_1d[:-1]
    down_move = low_1d[:-1] - low_1d[1:]
    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0.0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0.0)
    plus_dm = np.concatenate([[np.nan], plus_dm])
    minus_dm = np.concatenate([[np.nan], minus_dm])
    
    # Smooth TR, +DM, -DM using Wilder's smoothing (equivalent to EMA with alpha=1/14)
    def wilder_smooth(data, period):
        result = np.full_like(data, np.nan)
        if len(data) < period:
            return result
        # First value is simple average
        result[period-1] = np.nanmean(data[:period])
        # Subsequent values: smoothed = prev_smoothed - (prev_smoothed/period) + current
        for i in range(period, len(data)):
            if not np.isnan(result[i-1]) and not np.isnan(data[i]):
                result[i] = result[i-1] - (result[i-1]/period) + data[i]
        return result
    
    atr = wilder_smooth(tr, 14)
    plus_di = 100 * wilder_smooth(plus_dm, 14) / atr
    minus_di = 100 * wilder_smooth(minus_dm, 14) / atr
    dx = 100 * np.abs(plus_di - minus_di) / (plus_di + minus_di)
    adx = wilder_smooth(dx, 14)
    
    # Align 1d ADX to 6h timeframe
    adx_aligned = align_htf_to_ltf(prices, df_1d, adx)
    
    # 1w regime: close > SMA50 for bull regime, close < SMA50 for bear regime
    close_1w = df_1w['close'].values
    sma_50_1w = pd.Series(close_1w).rolling(window=50, min_periods=50).mean().values
    bull_regime = close_1w > sma_50_1w
    bear_regime = close_1w < sma_50_1w
    
    # Align 1w regime to 6h timeframe
    bull_regime_aligned = align_htf_to_ltf(prices, df_1w, bull_regime.astype(float))
    bear_regime_aligned = align_htf_to_ltf(prices, df_1w, bear_regime.astype(float))
    
    # Donchian(20) channels on 6h
    donchian_high = pd.Series(high).rolling(window=20, min_periods=20).max().values
    donchian_low = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # Volume confirmation: current 6h volume > 2.0x 20-bar average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 60  # warmup for Donchian, ADX, volume MA
    
    for i in range(start_idx, n):
        if np.isnan(donchian_high[i]) or np.isnan(donchian_low[i]) or np.isnan(adx_aligned[i]) or np.isnan(vol_ma[i]) or np.isnan(bull_regime_aligned[i]) or np.isnan(bear_regime_aligned[i]):
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
            
        volume_confirm = curr_vol > (curr_vol_ma * 2.0)  # Volume spike threshold
        
        # Donchian breakout signals
        breakout_up = curr_high > donchian_high[i]  # break above upper band
        breakout_down = curr_low < donchian_low[i]  # break below lower band
        
        # Entry conditions
        if position == 0:  # Flat - look for new entries
            # Long: breakout above Donchian high AND 1d ADX > 25 AND bull regime AND volume confirmation
            if (breakout_up and 
                adx_aligned[i] > 25 and 
                bull_regime_aligned[i] > 0.5 and 
                volume_confirm):
                signals[i] = 0.25
                position = 1
            # Short: breakout below Donchian low AND 1d ADX > 25 AND bear regime AND volume confirmation
            elif (breakout_down and 
                  adx_aligned[i] > 25 and 
                  bear_regime_aligned[i] > 0.5 and 
                  volume_confirm):
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        
        elif position == 1:  # Long position
            # Exit: price crosses below Donchian low (stoploss) OR ADX < 20 (trend weakening) OR regime change
            if (curr_low < donchian_low[i] or 
                adx_aligned[i] < 20 or 
                bull_regime_aligned[i] <= 0.5):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:  # Short position
            # Exit: price crosses above Donchian high (stoploss) OR ADX < 20 (trend weakening) OR regime change
            if (curr_high > donchian_high[i] or 
                adx_aligned[i] < 20 or 
                bear_regime_aligned[i] <= 0.5):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals
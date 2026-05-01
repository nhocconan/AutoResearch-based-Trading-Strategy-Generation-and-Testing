#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Donchian(20) breakout with 1d trend filter (EMA34) and volume confirmation.
# Long when price breaks above Donchian(20) high AND 1d EMA34 > 1d EMA89 (bullish trend) AND volume > 1.8x 20-bar average.
# Short when price breaks below Donchian(20) low AND 1d EMA34 < 1d EMA89 (bearish trend) AND volume > 1.8x 20-bar average.
# Uses discrete sizing 0.25 to manage drawdown. Target: 75-200 total trades over 4 years (19-50/year).
# 1d EMA trend filter ensures alignment with higher timeframe momentum, reducing false breakouts in ranging markets.
# Volume confirmation set to 1.8x to avoid choppy market noise while capturing institutional participation.
# Primary timeframe: 4h, HTF: 1d for trend and structure.

name = "4h_Donchian20_1dEMA_Trend_VolumeSpike_v1"
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
    
    # Load 1d data ONCE before loop for trend and structure
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 90:  # Need enough for EMA89
        return np.zeros(n)
    
    # 1d EMA trend: fast EMA34, slow EMA89
    ema_fast = pd.Series(df_1d['close'].values).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_slow = pd.Series(df_1d['close'].values).ewm(span=89, adjust=False, min_periods=89).mean().values
    trend_bullish = ema_fast > ema_slow  # 1 = bullish trend
    trend_bearish = ema_fast < ema_slow  # 1 = bearish trend
    
    # Donchian(20) channels from 1d data (more stable than 4h for structure)
    donchian_high = pd.Series(df_1d['high'].values).rolling(window=20, min_periods=20).max().shift(1).values
    donchian_low = pd.Series(df_1d['low'].values).rolling(window=20, min_periods=20).min().shift(1).values
    
    # Align 1d indicators to 4h timeframe
    trend_bullish_aligned = align_htf_to_ltf(prices, df_1d, trend_bullish.astype(float))
    trend_bearish_aligned = align_htf_to_ltf(prices, df_1d, trend_bearish.astype(float))
    donchian_high_aligned = align_htf_to_ltf(prices, df_1d, donchian_high)
    donchian_low_aligned = align_htf_to_ltf(prices, df_1d, donchian_low)
    
    # Volume confirmation: current 4h volume > 1.8x 20-bar average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 100  # warmup for Donchian and EMAs
    
    for i in range(start_idx, n):
        if np.isnan(donchian_high_aligned[i]) or np.isnan(donchian_low_aligned[i]) or \
           np.isnan(trend_bullish_aligned[i]) or np.isnan(trend_bearish_aligned[i]) or \
           np.isnan(vol_ma[i]):
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
            
        volume_confirm = curr_vol > (curr_vol_ma * 1.8)  # Volume spike threshold
        
        # Donchian breakout signals
        breakout_up = curr_high > donchian_high_aligned[i]  # break above Donchian high
        breakout_down = curr_low < donchian_low_aligned[i]  # break below Donchian low
        
        # Entry conditions
        if position == 0:  # Flat - look for new entries
            # Long: breakout above Donchian high AND bullish 1d trend AND volume confirmation
            if (breakout_up and 
                trend_bullish_aligned[i] == 1.0 and 
                volume_confirm):
                signals[i] = 0.25
                position = 1
            # Short: breakout below Donchian low AND bearish 1d trend AND volume confirmation
            elif (breakout_down and 
                  trend_bearish_aligned[i] == 1.0 and 
                  volume_confirm):
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        
        elif position == 1:  # Long position
            # Exit: price crosses below Donchian low (stoploss) OR 1d trend turns bearish
            if (curr_low < donchian_low_aligned[i] or 
                trend_bearish_aligned[i] == 1.0):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:  # Short position
            # Exit: price crosses above Donchian high (stoploss) OR 1d trend turns bullish
            if (curr_high > donchian_high_aligned[i] or 
                trend_bullish_aligned[i] == 1.0):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals
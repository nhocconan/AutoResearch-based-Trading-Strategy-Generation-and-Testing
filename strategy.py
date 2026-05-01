#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Donchian(20) breakout with 1d trend filter and volume confirmation.
# Long when price breaks above Donchian(20) high AND 1d EMA50 > EMA200 (bullish trend) AND volume > 1.5x 20-bar average.
# Short when price breaks below Donchian(20) low AND 1d EMA50 < EMA200 (bearish trend) AND volume > 1.5x 20-bar average.
# Uses discrete sizing 0.25 to manage drawdown. Target: 75-200 total trades over 4 years (19-50/year).
# 1d EMA crossover provides higher timeframe trend bias, reducing false breakouts in choppy markets.
# Volume confirmation threshold set to 1.5x to avoid excessive trades while capturing momentum bursts.
# Primary timeframe: 4h, HTF: 1d for trend filter.

name = "4h_Donchian20_1dEMATrend_VolumeConfirm_v1"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Load 1d data ONCE before loop for trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 200:
        return np.zeros(n)
    
    # 1d EMA50 and EMA200 for trend filter
    close_1d = pd.Series(df_1d['close'].values)
    ema50 = close_1d.ewm(span=50, adjust=False, min_periods=50).mean().values
    ema200 = close_1d.ewm(span=200, adjust=False, min_periods=200).mean().values
    
    # 1d trend: 1 = bullish (EMA50 > EMA200), -1 = bearish (EMA50 < EMA200), 0 = neutral
    trend_1d_raw = np.where(ema50 > ema200, 1, np.where(ema50 < ema200, -1, 0))
    trend_1d_aligned = align_htf_to_ltf(prices, df_1d, trend_1d_raw)
    
    # Calculate Donchian(20) channels from 4h data
    donchian_high = pd.Series(high).rolling(window=20, min_periods=20).max().shift(1).values
    donchian_low = pd.Series(low).rolling(window=20, min_periods=20).min().shift(1).values
    
    # Volume confirmation: current 4h volume > 1.5x 20-bar average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 50  # warmup for Donchian and indicators
    
    for i in range(start_idx, n):
        if np.isnan(donchian_high[i]) or np.isnan(donchian_low[i]) or \
           np.isnan(trend_1d_aligned[i]) or np.isnan(vol_ma[i]):
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
            
        volume_confirm = curr_vol > (curr_vol_ma * 1.5)  # Volume confirmation threshold
        
        # Donchian breakout signals
        breakout_up = curr_high > donchian_high[i]  # break above Donchian high
        breakout_down = curr_low < donchian_low[i]  # break below Donchian low
        
        # 1d trend filter
        bullish_trend = trend_1d_aligned[i] == 1
        bearish_trend = trend_1d_aligned[i] == -1
        
        # Entry conditions
        if position == 0:  # Flat - look for new entries
            # Long: breakout above Donchian high AND bullish 1d trend AND volume confirmation
            if (breakout_up and 
                bullish_trend and 
                volume_confirm):
                signals[i] = 0.25
                position = 1
            # Short: breakout below Donchian low AND bearish 1d trend AND volume confirmation
            elif (breakout_down and 
                  bearish_trend and 
                  volume_confirm):
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        
        elif position == 1:  # Long position
            # Exit: price crosses below Donchian low (stoploss) OR 1d trend turns bearish
            if (curr_low < donchian_low[i] or 
                trend_1d_aligned[i] == -1):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:  # Short position
            # Exit: price crosses above Donchian high (stoploss) OR 1d trend turns bullish
            if (curr_high > donchian_high[i] or 
                trend_1d_aligned[i] == 1):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals
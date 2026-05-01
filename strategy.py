#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Donchian(20) breakout with 12h trend filter and volume confirmation.
# Long when price breaks above Donchian(20) high AND 12h close > 12h open (bullish 12h) AND volume > 1.5x 20-bar average.
# Short when price breaks below Donchian(20) low AND 12h close < 12h open (bearish 12h) AND volume > 1.5x 20-bar average.
# Uses discrete sizing 0.25 to manage drawdown. Target: 75-200 total trades over 4 years (19-50/year).
# 12h trend filter reduces false breakouts by aligning with higher timeframe momentum.
# Volume spike threshold set to 1.5x to avoid choppy market noise while capturing institutional participation.
# Primary timeframe: 4h, HTF: 12h for trend bias.

name = "4h_Donchian20_12hTrend_VolumeSpike_v1"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 60:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Load 12h data ONCE before loop for trend filter
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 2:
        return np.zeros(n)
    
    # 12h trend: 1 = bullish 12h bar (close > open), -1 = bearish 12h bar (close < open)
    trend_12h_raw = np.where(df_12h['close'].values > df_12h['open'].values, 1,
                             np.where(df_12h['close'].values < df_12h['open'].values, -1, 0))
    trend_12h_aligned = align_htf_to_ltf(prices, df_12h, trend_12h_raw)
    
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
           np.isnan(trend_12h_aligned[i]) or np.isnan(vol_ma[i]):
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
        
        # Donchian breakout signals
        breakout_up = curr_high > donchian_high[i]  # break above Donchian high
        breakout_down = curr_low < donchian_low[i]  # break below Donchian low
        
        # 12h trend filter
        bullish_12h = trend_12h_aligned[i] == 1
        bearish_12h = trend_12h_aligned[i] == -1
        
        # Entry conditions
        if position == 0:  # Flat - look for new entries
            # Long: breakout above Donchian high AND bullish 12h AND volume confirmation
            if (breakout_up and 
                bullish_12h and 
                volume_confirm):
                signals[i] = 0.25
                position = 1
            # Short: breakout below Donchian low AND bearish 12h AND volume confirmation
            elif (breakout_down and 
                  bearish_12h and 
                  volume_confirm):
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        
        elif position == 1:  # Long position
            # Exit: price crosses below Donchian low (stoploss) OR 12h trend turns bearish
            if (curr_low < donchian_low[i] or 
                trend_12h_aligned[i] == -1):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:  # Short position
            # Exit: price crosses above Donchian high (stoploss) OR 12h trend turns bullish
            if (curr_high > donchian_high[i] or 
                trend_12h_aligned[i] == 1):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals
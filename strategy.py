#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Donchian(20) breakout with daily trend filter and volume confirmation.
# Long when price breaks above Donchian(20) high AND daily close > daily EMA(34) AND volume > 1.5x 20-bar average.
# Short when price breaks below Donchian(20) low AND daily close < daily EMA(34) AND volume > 1.5x 20-bar average.
# Uses discrete sizing 0.25 to manage drawdown. Target: 75-200 total trades over 4 years (19-50/year).
# Daily EMA(34) filter ensures alignment with higher timeframe trend, reducing false breakouts in choppy markets.
# Volume confirmation set to 1.5x to avoid overtrading while capturing momentum bursts.
# Primary timeframe: 4h, HTF: 1d for trend filter.

name = "4h_Donchian20_DailyEMA34_Trend_VolumeConfirm_v1"
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
    
    # Load 1d data ONCE before loop for daily trend and Donchian channels
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 35:  # Need 34 for EMA + 1 for shift
        return np.zeros(n)
    
    # Daily EMA(34) for trend filter
    ema_34 = pd.Series(df_1d['close'].values).ewm(span=34, adjust=False, min_periods=34).mean().values
    # Align EMA to 4h timeframe (completed 1d bar only)
    ema_34_aligned = align_htf_to_ltf(prices, df_1d, ema_34)
    
    # Donchian(20) channels from 1d high/low (more stable than 4h for structure)
    donchian_high = pd.Series(df_1d['high'].values).rolling(window=20, min_periods=20).max().shift(1).values
    donchian_low = pd.Series(df_1d['low'].values).rolling(window=20, min_periods=20).min().shift(1).values
    # Align Donchian levels to 4h timeframe
    donchian_high_aligned = align_htf_to_ltf(prices, df_1d, donchian_high)
    donchian_low_aligned = align_htf_to_ltf(prices, df_1d, donchian_low)
    
    # Volume confirmation: current 4h volume > 1.5x 20-bar average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 50  # warmup for Donchian and EMA
    
    for i in range(start_idx, n):
        if np.isnan(donchian_high_aligned[i]) or np.isnan(donchian_low_aligned[i]) or \
           np.isnan(ema_34_aligned[i]) or np.isnan(vol_ma[i]):
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
        breakout_up = curr_high > donchian_high_aligned[i]  # break above Donchian high
        breakout_down = curr_low < donchian_low_aligned[i]  # break below Donchian low
        
        # Daily trend filter
        bullish_trend = curr_close > ema_34_aligned[i]
        bearish_trend = curr_close < ema_34_aligned[i]
        
        # Entry conditions
        if position == 0:  # Flat - look for new entries
            # Long: breakout above Donchian high AND bullish daily trend AND volume confirmation
            if (breakout_up and 
                bullish_trend and 
                volume_confirm):
                signals[i] = 0.25
                position = 1
            # Short: breakout below Donchian low AND bearish daily trend AND volume confirmation
            elif (breakout_down and 
                  bearish_trend and 
                  volume_confirm):
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        
        elif position == 1:  # Long position
            # Exit: price crosses below Donchian low (stoploss) OR daily trend turns bearish
            if (curr_low < donchian_low_aligned[i] or 
                not bullish_trend):  # price below EMA(34)
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:  # Short position
            # Exit: price crosses above Donchian high (stoploss) OR daily trend turns bullish
            if (curr_high > donchian_high_aligned[i] or 
                not bearish_trend):  # price above EMA(34)
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals
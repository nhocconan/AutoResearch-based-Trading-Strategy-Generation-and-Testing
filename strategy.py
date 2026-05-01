#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h Donchian(20) breakout with daily trend filter and volume confirmation.
# Long when price breaks above Donchian(20) high AND daily close > daily EMA34 (bullish trend) AND volume > 2.0x 20-bar average.
# Short when price breaks below Donchian(20) low AND daily close < daily EMA34 (bearish trend) AND volume > 2.0x 20-bar average.
# Uses discrete sizing 0.25 to manage drawdown. Target: 50-150 total trades over 4 years (12-37/year).
# Daily EMA34 filter ensures alignment with higher timeframe trend, reducing false breakouts in choppy markets.
# Volume spike threshold set to 2.0x to avoid low-momentum noise.
# Primary timeframe: 12h, HTF: 1d for trend filter.

name = "12h_Donchian20_DailyEMA34_Trend_VolumeSpike_v1"
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
    
    # Load 1d data ONCE before loop for daily trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 35:  # need at least 34 for EMA + 1 for shift
        return np.zeros(n)
    
    # Daily EMA34 trend: 1 = bullish (close > EMA34), -1 = bearish (close < EMA34)
    close_1d = df_1d['close'].values
    ema_34 = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    daily_trend_raw = np.where(close_1d > ema_34, 1,
                               np.where(close_1d < ema_34, -1, 0))
    daily_trend_aligned = align_htf_to_ltf(prices, df_1d, daily_trend_raw)
    
    # Calculate Donchian(20) channels from 1d data (more stable than 12h for structure)
    if len(df_1d) < 20:
        return np.zeros(n)
    
    # Donchian high: max(high, 20) from previous completed day
    donchian_high = pd.Series(df_1d['high'].values).rolling(window=20, min_periods=20).max().shift(1).values
    # Donchian low: min(low, 20) from previous completed day
    donchian_low = pd.Series(df_1d['low'].values).rolling(window=20, min_periods=20).min().shift(1).values
    
    # Align Donchian levels to 12h timeframe
    donchian_high_aligned = align_htf_to_ltf(prices, df_1d, donchian_high)
    donchian_low_aligned = align_htf_to_ltf(prices, df_1d, donchian_low)
    
    # Volume confirmation: current 12h volume > 2.0x 20-bar average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 55  # warmup for Donchian (20) + EMA (34) + volume MA (20)
    
    for i in range(start_idx, n):
        if np.isnan(donchian_high_aligned[i]) or np.isnan(donchian_low_aligned[i]) or \
           np.isnan(daily_trend_aligned[i]) or np.isnan(vol_ma[i]):
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
        breakout_up = curr_high > donchian_high_aligned[i]  # break above Donchian high
        breakout_down = curr_low < donchian_low_aligned[i]  # break below Donchian low
        
        # Daily trend filter
        bullish_trend = daily_trend_aligned[i] == 1
        bearish_trend = daily_trend_aligned[i] == -1
        
        # Entry conditions
        if position == 0:  # Flat - look for new entries
            # Long: breakout above Donchian high AND bullish trend AND volume confirmation
            if (breakout_up and 
                bullish_trend and 
                volume_confirm):
                signals[i] = 0.25
                position = 1
            # Short: breakout below Donchian low AND bearish trend AND volume confirmation
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
                daily_trend_aligned[i] == -1):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:  # Short position
            # Exit: price crosses above Donchian high (stoploss) OR daily trend turns bullish
            if (curr_high > donchian_high_aligned[i] or 
                daily_trend_aligned[i] == 1):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals
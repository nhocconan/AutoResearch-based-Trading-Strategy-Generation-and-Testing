#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1d Donchian(20) breakout with 1w EMA34 trend filter and volume confirmation.
# Long when price breaks above Donchian upper band AND 1w close > EMA34 (bullish trend) AND volume > 1.8x 20-bar average.
# Short when price breaks below Donchian lower band AND 1w close < EMA34 (bearish trend) AND volume > 1.8x 20-bar average.
# Uses discrete sizing 0.25 to manage drawdown. Target: 30-100 total trades over 4 years (7-25/year).
# Donchian channels provide clear breakout levels, 1w EMA34 filters higher-timeframe trend, volume spike confirms momentum.
# Primary timeframe: 1d, HTF: 1w for EMA trend filter.

name = "1d_Donchian20_1wEMA34_Trend_VolumeSpike_v1"
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
    
    # Load 1w data ONCE before loop for EMA trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 34:
        return np.zeros(n)
    
    # Calculate Donchian(20) channels
    highest_high = pd.Series(high).rolling(window=20, min_periods=20).max().values
    lowest_low = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # 1w EMA34 trend filter
    prev_close = df_1w['close'].values
    ema_34 = pd.Series(prev_close).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_aligned = align_htf_to_ltf(prices, df_1w, ema_34)
    
    # Volume confirmation: current 1d volume > 1.8x 20-bar average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 34  # warmup for EMA34 and Donchian
    
    for i in range(start_idx, n):
        if np.isnan(highest_high[i]) or np.isnan(lowest_low[i]) or np.isnan(ema_34_aligned[i]) or np.isnan(vol_ma[i]):
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
        
        # Trend filter: bullish if close > EMA34, bearish if close < EMA34
        bullish_trend = curr_close > ema_34_aligned[i]
        bearish_trend = curr_close < ema_34_aligned[i]
        
        # Entry conditions
        if position == 0:  # Flat - look for new entries
            # Long: price breaks above upper band AND bullish trend AND volume confirmation
            if (curr_high > highest_high[i] and 
                bullish_trend and 
                volume_confirm):
                signals[i] = 0.25
                position = 1
            # Short: price breaks below lower band AND bearish trend AND volume confirmation
            elif (curr_low < lowest_low[i] and 
                  bearish_trend and 
                  volume_confirm):
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        
        elif position == 1:  # Long position
            # Exit: price crosses below lower band (breakdown) OR trend turns bearish
            if (curr_low < lowest_low[i] or 
                bearish_trend):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:  # Short position
            # Exit: price crosses above upper band (breakout) OR trend turns bullish
            if (curr_high > highest_high[i] or 
                bullish_trend):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals
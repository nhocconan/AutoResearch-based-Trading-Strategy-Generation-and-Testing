#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1d Donchian(20) breakout with 1w EMA50 trend filter and volume confirmation.
# Uses 1w EMA50 > EMA200 to identify strong bull/bear regimes, reducing whipsaws.
# Long when price breaks above Donchian upper AND 1w EMA50 > EMA200 AND volume > 2.0x 20-bar average.
# Short when price breaks below Donchian lower AND 1w EMA50 < EMA200 AND volume > 2.0x 20-bar average.
# Uses discrete sizing 0.25 to manage drawdown. Target: 30-100 total trades over 4 years.
# Volume spike threshold set to 2.0x to ensure high-conviction entries.

name = "1d_Donchian20_1wEMA50_200_Trend_VolumeSpike_v1"
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
    if len(df_1w) < 50:  # Need enough for EMA200 calculation
        return np.zeros(n)
    
    # 1w EMA50 and EMA200 calculation
    close_1w = df_1w['close'].values
    ema50_1w = pd.Series(close_1w).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema200_1w = pd.Series(close_1w).ewm(span=200, adjust=False, min_periods=200).mean().values
    
    # Align 1w EMAs to 1d timeframe
    ema50_aligned = align_htf_to_ltf(prices, df_1w, ema50_1w)
    ema200_aligned = align_htf_to_ltf(prices, df_1w, ema200_1w)
    
    # 1w trend: EMA50 > EMA200 for bull, EMA50 < EMA200 for bear
    bull_trend = ema50_aligned > ema200_aligned
    bear_trend = ema50_aligned < ema200_aligned
    
    # Donchian(20) channels on 1d (using previous 20 bars)
    high_roll = pd.Series(high).rolling(window=20, min_periods=20).max().shift(1).values
    low_roll = pd.Series(low).rolling(window=20, min_periods=20).min().shift(1).values
    
    # Volume confirmation: current 1d volume > 2.0x 20-bar average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 50  # warmup for Donchian and volume MA
    
    for i in range(start_idx, n):
        if np.isnan(high_roll[i]) or np.isnan(low_roll[i]) or np.isnan(ema50_aligned[i]) or np.isnan(ema200_aligned[i]) or np.isnan(vol_ma[i]):
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
        breakout_up = curr_high > high_roll[i]  # break above upper channel
        breakout_down = curr_low < low_roll[i]  # break below lower channel
        
        # Entry conditions
        if position == 0:  # Flat - look for new entries
            # Long: breakout above upper AND bull trend AND volume confirmation
            if (breakout_up and 
                bull_trend[i] and 
                volume_confirm):
                signals[i] = 0.25
                position = 1
            # Short: breakout below lower AND bear trend AND volume confirmation
            elif (breakout_down and 
                  bear_trend[i] and 
                  volume_confirm):
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        
        elif position == 1:  # Long position
            # Exit: price crosses below lower channel OR trend reversal
            if (curr_low < low_roll[i] or 
                ema50_aligned[i] < ema200_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:  # Short position
            # Exit: price crosses above upper channel OR trend reversal
            if (curr_high > high_roll[i] or 
                ema50_aligned[i] > ema200_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals
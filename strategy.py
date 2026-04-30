#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h strategy using 1d Williams %R extreme levels with 12h EMA(50) trend filter and volume confirmation
# Williams %R identifies overbought/oversold conditions: > -20 = overbought, < -80 = oversold
# In bull markets: buy when %R crosses above -80 from below (oversold bounce) with uptrend
# In bear markets: sell when %R crosses below -20 from above (overbought rejection) with downtrend
# Volume confirmation ensures institutional participation. Designed for low trade frequency (~20-40/year on 4h).
# Uses 1d HTF for Williams %R (institutional extreme levels) and 12h HTF for EMA trend filter.
# Works in both bull and bear markets by fading extremes with trend alignment.

name = "4h_1dWilliamsR_Extreme_12hEMA50_VolumeConfirm_v1"
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
    
    # Load 1d data ONCE before loop for Williams %R calculation
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 14:
        return np.zeros(n)
    
    # Calculate 14-period Williams %R: (Highest High - Close) / (Highest High - Lowest Low) * -100
    highest_high = pd.Series(df_1d['high'].values).rolling(window=14, min_periods=14).max().values
    lowest_low = pd.Series(df_1d['low'].values).rolling(window=14, min_periods=14).min().values
    williams_r = -100 * (highest_high - df_1d['close'].values) / (highest_high - lowest_low + 1e-10)
    
    # Align 1d Williams %R to 4h timeframe (wait for 1d bar to close)
    williams_r_aligned = align_htf_to_ltf(prices, df_1d, williams_r)
    
    # Load 12h data ONCE before loop for EMA(50) trend filter
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 50:
        return np.zeros(n)
    
    # Calculate 12h EMA(50) for trend filter
    close_12h_s = pd.Series(df_12h['close'].values)
    ema_50_12h = close_12h_s.ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_12h_aligned = align_htf_to_ltf(prices, df_12h, ema_50_12h)
    
    # Calculate ATR(14) for dynamic stoploss
    tr1 = high[1:] - low[1:]
    tr2 = np.abs(high[1:] - close[:-1])
    tr3 = np.abs(low[1:] - close[:-1])
    tr = np.concatenate([[np.max([tr1[0], tr2[0], tr3[0]])], np.maximum(tr1, np.maximum(tr2, tr3))])
    atr = pd.Series(tr).ewm(span=14, adjust=False, min_periods=14).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    start_idx = 60  # warmup for EMA(50) and Williams %R
    
    for i in range(start_idx, n):
        # Volume confirmation: volume > 1.8x 30-period average (stricter to reduce trades)
        if i >= 30:
            vol_ma_30 = np.mean(volume[i-30:i])
        elif i > 0:
            vol_ma_30 = np.mean(volume[:i])
        else:
            vol_ma_30 = 0
        volume_spike = volume[i] > (1.8 * vol_ma_30) if vol_ma_30 > 0 else False
        
        curr_close = close[i]
        curr_high = high[i]
        curr_low = low[i]
        curr_ema = ema_50_12h_aligned[i]
        curr_atr = atr[i]
        curr_williams_r = williams_r_aligned[i]
        
        if position == 0:  # Flat - look for new entries
            # Require volume spike and Williams %R extreme with trend alignment
            if volume_spike:
                # Bullish entry: Williams %R crosses above -80 from below (oversold bounce) with 12h uptrend
                if curr_williams_r > -80 and curr_williams_r < -20 and curr_close > curr_ema:
                    # Additional confirmation: previous bar was below -80
                    if i > start_idx and williams_r_aligned[i-1] <= -80:
                        signals[i] = 0.25
                        position = 1
                        entry_price = curr_close
                # Bearish entry: Williams %R crosses below -20 from above (overbought rejection) with 12h downtrend
                elif curr_williams_r < -20 and curr_williams_r > -80 and curr_close < curr_ema:
                    # Additional confirmation: previous bar was above -20
                    if i > start_idx and williams_r_aligned[i-1] >= -20:
                        signals[i] = -0.25
                        position = -1
                        entry_price = curr_close
        
        elif position == 1:  # Long position
            # Stoploss: 2.5 * ATR below entry price OR Williams %R reaches overbought (-20)
            if curr_close < entry_price - 2.5 * curr_atr:
                signals[i] = 0.0
                position = 0
            elif curr_williams_r >= -20:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:  # Short position
            # Stoploss: 2.5 * ATR above entry price OR Williams %R reaches oversold (-80)
            if curr_close > entry_price + 2.5 * curr_atr:
                signals[i] = 0.0
                position = 0
            elif curr_williams_r <= -80:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals
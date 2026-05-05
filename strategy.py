#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h Donchian(20) breakout with 1d EMA34 trend filter and volume confirmation
# Long when price breaks above upper band AND close > EMA34(1d) AND volume > 1.8x 20-period average
# Short when price breaks below lower band AND close < EMA34(1d) AND volume > 1.8x 20-period average
# Exit when price crosses back to the opposite Donchian band OR EMA34(1d) trend flips
# Uses 12h primary timeframe for low trade frequency (target: 12-37/year) and 1d HTF for trend alignment
# Discrete sizing (0.25) to limit fee drag and manage drawdown in both bull and bear markets

name = "12h_Donchian20_1dEMA34_Trend_Volume"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Get 1d data ONCE before loop for EMA34 trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    # Calculate EMA34 on 1d close for trend filter
    ema_34_1d = pd.Series(df_1d['close'].values).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # Calculate 12h Donchian channels (20-period) from prior bar to avoid look-ahead
    # Shift by 1 to use only completed bars
    prev_high = np.roll(high, 1)
    prev_low = np.roll(low, 1)
    prev_high[0] = np.nan
    prev_low[0] = np.nan
    
    # Calculate rolling max/min on prior bars
    high_series = pd.Series(prev_high)
    low_series = pd.Series(prev_low)
    donchian_high = high_series.rolling(window=20, min_periods=20).max().values
    donchian_low = low_series.rolling(window=20, min_periods=20).min().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(50, n):
        # Skip if any value is NaN
        if (np.isnan(donchian_high[i]) or 
            np.isnan(donchian_low[i]) or 
            np.isnan(ema_34_1d_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Volume confirmation: volume > 1.8x 20-period average
        if i >= 20:
            vol_ma_20 = pd.Series(volume[:i+1]).rolling(window=20, min_periods=20).mean().iloc[-1]
            volume_filter = volume[i] > (1.8 * vol_ma_20)
        else:
            volume_filter = False
        
        if position == 0:
            # Long conditions: price breaks above upper band AND close > EMA34(1d) AND volume spike
            if (close[i] > donchian_high[i] and 
                close[i] > ema_34_1d_aligned[i] and 
                volume_filter):
                signals[i] = 0.25
                position = 1
            # Short conditions: price breaks below lower band AND close < EMA34(1d) AND volume spike
            elif (close[i] < donchian_low[i] and 
                  close[i] < ema_34_1d_aligned[i] and 
                  volume_filter):
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: price crosses below lower band OR close < EMA34(1d) (trend flip)
            if (close[i] < donchian_low[i] or 
                close[i] < ema_34_1d_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: price crosses above upper band OR close > EMA34(1d) (trend flip)
            if (close[i] > donchian_high[i] or 
                close[i] > ema_34_1d_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals
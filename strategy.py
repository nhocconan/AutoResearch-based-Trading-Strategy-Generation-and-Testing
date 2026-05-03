#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Donchian(20) breakout with 1d EMA50 trend filter and volume confirmation.
# Long when price breaks above 4h Donchian high in bull trend (close > 1d EMA50) with volume > 1.8x 20-period MA.
# Short when price breaks below 4h Donchian low in bear trend (close < 1d EMA50) with volume spike.
# Uses discrete position sizing (0.30) to balance return and drawdown. Donchian channels provide
# clear trend-following structure proven on SOLUSDT. 1d trend filter reduces whipsaw vs shorter MAs.
# Volume confirmation ensures institutional participation. Target: 75-200 total trades over 4 years (19-50/year).

name = "4h_Donchian20_1dEMA50_Volume"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Get 1d data for EMA trend filter
    df_1d = get_htf_data(prices, '1d')
    
    if len(df_1d) < 2:
        return np.zeros(n)
    
    # Calculate 1d EMA50 for trend filter
    ema_50_1d = pd.Series(df_1d['close'].values).ewm(span=50, min_periods=50, adjust=False).mean().values
    
    # Align EMA to 4h timeframe
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # Donchian channels (20-period) on 4h
    donchian_high = pd.Series(high).rolling(window=20, min_periods=20).max().values
    donchian_low = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # Volume regime: current 4h volume > 1.8x 20-period MA
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (1.8 * vol_ma_20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(100, n):
        # Skip if any value is NaN
        if (np.isnan(ema_50_1d_aligned[i]) or np.isnan(donchian_high[i]) or 
            np.isnan(donchian_low[i]) or np.isnan(vol_ma_20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
            
        close_val = close[i]
        ema_trend = ema_50_1d_aligned[i]
        upper_channel = donchian_high[i]
        lower_channel = donchian_low[i]
        vol_spike = volume_spike[i]
        
        # Determine trend regime
        is_bull_trend = close_val > ema_trend
        is_bear_trend = close_val < ema_trend
        
        # Donchian breakout conditions
        breakout_up = close_val > upper_channel
        breakout_down = close_val < lower_channel
        
        # Entry logic
        if position == 0:
            if is_bull_trend and breakout_up and vol_spike:
                signals[i] = 0.30
                position = 1
            elif is_bear_trend and breakout_down and vol_spike:
                signals[i] = -0.30
                position = -1
        elif position == 1:
            # Long exit: Donchian low break OR trend reversal
            if close_val < lower_channel or close_val < ema_trend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.30
        elif position == -1:
            # Short exit: Donchian high break OR trend reversal
            if close_val > upper_channel or close_val > ema_trend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.30
    
    return signals
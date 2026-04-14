#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1d Donchian breakout with weekly trend filter and volume confirmation
# Targets: 15-25 trades/year by trading breakouts in the direction of weekly trend
# Logic: Long when price breaks above Donchian(20) high and weekly trend is up
#        Short when price breaks below Donchian(20) low and weekly trend is down
#        Exit when price crosses opposite Donchian level or trend reverses
# Position size: 0.25 to manage drawdown

def generate_signals(prices):
    n = len(prices)
    if n < 60:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Load weekly data once
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 20:
        return np.zeros(n)
    
    # Weekly EMA50 for trend filter
    close_1w = df_1w['close'].values
    ema_50_1w = pd.Series(close_1w).ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Donchian channels (20-period)
    high_20 = pd.Series(high).rolling(window=20, min_periods=20).max().values
    low_20 = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # Volume moving average for confirmation
    vol_ma = pd.Series(volume).ewm(span=20, adjust=False, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0
    position_size = 0.25
    
    for i in range(20, n):
        # Get aligned weekly EMA50
        ema_50_i = align_htf_to_ltf(prices, df_1w, ema_50_1w)[i]
        
        if np.isnan(high_20[i]) or np.isnan(low_20[i]) or np.isnan(ema_50_i) or np.isnan(vol_ma[i]):
            continue
        
        # Volume confirmation (1.5x average)
        volume_confirm = volume[i] > 1.5 * vol_ma[i]
        
        # Long: Break above Donchian high in uptrend with volume confirmation
        if position == 0 and close[i] > high_20[i] and close[i] > ema_50_i and volume_confirm:
            position = 1
            signals[i] = position_size
        # Short: Break below Donchian low in downtrend with volume confirmation
        elif position == 0 and close[i] < low_20[i] and close[i] < ema_50_i and volume_confirm:
            position = -1
            signals[i] = -position_size
        # Exit: Price crosses opposite Donchian level or trend reverses
        elif position != 0:
            if position == 1 and (close[i] < low_20[i] or close[i] < ema_50_i):
                position = 0
                signals[i] = 0.0
            elif position == -1 and (close[i] > high_20[i] or close[i] > ema_50_i):
                position = 0
                signals[i] = 0.0
    
    return signals

name = "1d_Donchian_WeeklyTrend_Volume"
timeframe = "1d"
leverage = 1.0
#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Donchian(20) breakout + 1d EMA34 trend filter + volume confirmation.
# Long when price breaks above 20-period high AND close > 1d EMA34 AND volume > 1.5x 20-period MA.
# Short when price breaks below 20-period low AND close < 1d EMA34 AND volume > 1.5x 20-period MA.
# Uses discrete sizing 0.25 to minimize fee churn. Target: 75-200 total trades over 4 years.
# Works in bull via long breakouts and bear via short breakdowns when aligned with 1d trend.

name = "4h_Donchian20_1dEMA34_Volume"
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
    
    if len(df_1d) < 34:
        return np.zeros(n)
    
    # Calculate 1d EMA34
    ema_34_1d = pd.Series(df_1d['close'].values).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # Donchian channels (20-period) on 4h
    donchian_high = pd.Series(high).rolling(window=20, min_periods=20).max().values
    donchian_low = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # Volume regime: current 4h volume > 1.5x 20-period MA
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (1.5 * vol_ma_20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(100, n):
        # Skip if any value is NaN
        if (np.isnan(ema_34_aligned[i]) or np.isnan(donchian_high[i]) or np.isnan(donchian_low[i]) or 
            np.isnan(vol_ma_20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
            
        close_val = close[i]
        ema_34_val = ema_34_aligned[i]
        vol_spike = volume_spike[i]
        
        # Entry logic
        if position == 0:
            # Long: break above Donchian high AND above 1d EMA34 AND volume spike
            if close_val > donchian_high[i] and close_val > ema_34_val and vol_spike:
                signals[i] = 0.25
                position = 1
            # Short: break below Donchian low AND below 1d EMA34 AND volume spike
            elif close_val < donchian_low[i] and close_val < ema_34_val and vol_spike:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Long exit: break below Donchian low OR close below 1d EMA34 OR volume drops
            if close_val < donchian_low[i] or close_val < ema_34_val or not vol_spike:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: break above Donchian high OR close above 1d EMA34 OR volume drops
            if close_val > donchian_high[i] or close_val > ema_34_val or not vol_spike:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals
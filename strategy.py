#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h Donchian(20) breakout + 1d volume spike + ADX regime filter
# Long when price breaks above 20-period Donchian high + volume > 1.5x 20-period EMA + ADX > 25
# Short when price breaks below 20-period Donchian low + volume > 1.5x 20-period EMA + ADX > 25
# Exit when price touches opposite Donchian band or ADX < 20
# Designed for 12h timeframe targeting 50-150 total trades over 4 years.
# Uses discrete position sizing (0.30) to balance return and drawdown.

name = "12h_Donchian20_1dVolumeSpike_ADXRegime"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Get 1d data for volume and ADX
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    # Calculate 1d volume EMA (20-period)
    vol_ema_20 = pd.Series(df_1d['volume']).ewm(span=20, adjust=False, min_periods=20).mean().values
    
    # Calculate 1d ADX (14-period)
    plus_dm = pd.Series(df_1d['high']).diff()
    minus_dm = pd.Series(df_1d['low']).diff().mul(-1)
    plus_dm = plus_dm.where((plus_dm > minus_dm) & (plus_dm > 0), 0.0)
    minus_dm = minus_dm.where((minus_dm > plus_dm) & (minus_dm > 0), 0.0)
    tr1 = pd.Series(df_1d['high']).sub(df_1d['low'])
    tr2 = pd.Series(df_1d['high']).sub(df_1d['close'].shift(1)).abs()
    tr3 = pd.Series(df_1d['low']).sub(df_1d['close'].shift(1)).abs()
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr = tr.rolling(window=14, min_periods=14).mean()
    plus_di = 100 * (plus_dm.rolling(window=14, min_periods=14).sum() / atr)
    minus_di = 100 * (minus_dm.rolling(window=14, min_periods=14).sum() / atr)
    dx = (abs(plus_di - minus_di) / (plus_di + minus_di)) * 100
    adx = dx.rolling(window=14, min_periods=14).mean()
    
    # Align 1d indicators to 12h timeframe
    vol_ema_20_aligned = align_htf_to_ltf(prices, df_1d, vol_ema_20)
    adx_aligned = align_htf_to_ltf(prices, df_1d, adx.values)
    
    # Calculate 12h Donchian channels (20-period)
    highest_high = pd.Series(high).rolling(window=20, min_periods=20).max().values
    lowest_low = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(100, n):
        # Skip if any value is NaN
        if (np.isnan(highest_high[i]) or np.isnan(lowest_low[i]) or 
            np.isnan(vol_ema_20_aligned[i]) or np.isnan(adx_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Volume confirmation: current 12h volume > 1.5x 1d volume EMA
        # Note: comparing 12h volume to 1d EMA volume (scaled)
        volume_confirm = volume[i] > (1.5 * vol_ema_20_aligned[i])
        
        if position == 0:
            # Only trade in strong trending regimes (ADX > 25)
            if adx_aligned[i] > 25:
                # Long breakout
                if close[i] > highest_high[i] and volume_confirm:
                    signals[i] = 0.30
                    position = 1
                # Short breakout
                elif close[i] < lowest_low[i] and volume_confirm:
                    signals[i] = -0.30
                    position = -1
        elif position == 1:
            # Exit long: price touches lower Donchian band OR ADX weakens (<20)
            if close[i] < lowest_low[i] or adx_aligned[i] < 20:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.30
        elif position == -1:
            # Exit short: price touches upper Donchian band OR ADX weakens (<20)
            if close[i] > highest_high[i] or adx_aligned[i] < 20:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.30
    
    return signals
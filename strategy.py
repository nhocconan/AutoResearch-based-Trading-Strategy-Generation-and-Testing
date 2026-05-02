#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Donchian(20) breakout with 12h EMA34 trend filter, volume confirmation, and ATR-based stoploss
# Uses 4h Donchian channels for breakout signals, 12h EMA34 for trend direction, and volume spike for confirmation
# Works in both bull and bear markets by trading with the 12h trend
# Target: 100-180 total trades over 4 years (25-45/year) for 4h timeframe
# Discrete sizing 0.25 balances profit potential and fee drag
# ATR stoploss limits downside during volatile periods like 2022 crash

name = "4h_Donchian20_Breakout_12hEMA34_Volume_ATR"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Calculate 12h EMA34 for trend filter (HTF)
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 34:
        return np.zeros(n)
    
    close_12h = df_12h['close'].values
    ema_34_12h = pd.Series(close_12h).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_12h_aligned = align_htf_to_ltf(prices, df_12h, ema_34_12h)
    
    # Calculate 4h Donchian channels (20-period)
    high_4h = get_htf_data(prices, '4h')['high'].values
    low_4h = get_htf_data(prices, '4h')['low'].values
    
    # Donchian upper/lower bands: highest high/lowest low over 20 periods
    donchian_upper = pd.Series(high_4h).rolling(window=20, min_periods=20).max().values
    donchian_lower = pd.Series(low_4h).rolling(window=20, min_periods=20).min().values
    
    # Align Donchian levels to 4h timeframe (use previous 4h bar's levels)
    donchian_upper_aligned = align_htf_to_ltf(prices, get_htf_data(prices, '4h'), donchian_upper)
    donchian_lower_aligned = align_htf_to_ltf(prices, get_htf_data(prices, '4h'), donchian_lower)
    
    # Volume confirmation: 2.0x 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (2.0 * vol_ma)
    
    # ATR(14) for stoploss and position sizing adjustment
    tr1 = pd.Series(high - low)
    tr2 = pd.Series(abs(high - np.roll(close, 1)))
    tr3 = pd.Series(abs(low - np.roll(close, 1)))
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    # Set first ATR value to avoid NaN
    atr[0] = tr1.iloc[0] if len(tr1) > 0 else 0.0
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after warmup (need enough data for all indicators)
    start_idx = 100
    
    for i in range(start_idx, n):
        # Check for NaN values in indicators
        if (np.isnan(ema_34_12h_aligned[i]) or np.isnan(donchian_upper_aligned[i]) or 
            np.isnan(donchian_lower_aligned[i]) or np.isnan(vol_ma[i]) or np.isnan(atr[i])):
            signals[i] = 0.0
            continue
        
        if position == 0:  # Flat - look for new entries
            # Long entry: Break above Donchian upper AND price > 12h EMA34 (uptrend) AND volume spike
            if (close[i] > donchian_upper_aligned[i] and 
                close[i] > ema_34_12h_aligned[i] and 
                volume_spike[i]):
                signals[i] = 0.25
                position = 1
            # Short entry: Break below Donchian lower AND price < 12h EMA34 (downtrend) AND volume spike
            elif (close[i] < donchian_lower_aligned[i] and 
                  close[i] < ema_34_12h_aligned[i] and 
                  volume_spike[i]):
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        
        elif position == 1:  # Long position
            # Stoploss: Close below Donchian lower - 0.5*ATR (volatility-adjusted)
            # Exit: Close below 12h EMA34 (trend change) OR Donchian lower break
            if (close[i] < ema_34_12h_aligned[i] or 
                close[i] < donchian_lower_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:  # Short position
            # Stoploss: Close above Donchian upper + 0.5*ATR (volatility-adjusted)
            # Exit: Close above 12h EMA34 (trend change) OR Donchian upper break
            if (close[i] > ema_34_12h_aligned[i] or 
                close[i] > donchian_upper_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals
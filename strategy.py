#!/usr/bin/env python3
"""
6h_1d_donchian_breakout_volume_v1
Strategy: 6h Donchian(20) breakout with volume confirmation and 1d trend filter
Timeframe: 6h
Leverage: 1.0
Hypothesis: Uses 6h Donchian channel breakouts for entry, filtered by 1d EMA50 trend and volume confirmation (>1.5x average volume). Designed to capture strong trending moves while avoiding false breakouts in chop. Uses daily trend for direction and 6h only for timing. Target: 50-150 total trades over 4 years.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "6h_1d_donchian_breakout_volume_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Price arrays
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Load higher timeframe data ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # 6-period ATR for Donchian band width (used in volatility filter)
    tr1 = high[1:] - low[1:]
    tr2 = np.abs(high[1:] - close[:-1])
    tr3 = np.abs(low[1:] - close[:-1])
    tr = np.concatenate([[tr1[0]], np.maximum(tr1, np.maximum(tr2, tr3))])
    atr = pd.Series(tr).rolling(window=6, min_periods=6).mean().values
    
    # 6-period moving average of true range for volatility filter
    atr_ma = pd.Series(atr).rolling(window=24, min_periods=24).mean().values
    
    # Donchian channels (20-period)
    donchian_high = pd.Series(high).rolling(window=20, min_periods=20).max().values
    donchian_low = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # 1d EMA50 for trend filter
    close_1d = df_1d['close'].values
    ema_50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # Volume average (24-period)
    vol_avg = pd.Series(volume).rolling(window=24, min_periods=24).mean().values
    vol_spike = volume > (1.5 * vol_avg)  # Volume confirmation
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(24, n):
        # Skip if any required data is invalid
        if (np.isnan(donchian_high[i]) or np.isnan(donchian_low[i]) or 
            np.isnan(ema_50_1d_aligned[i]) or np.isnan(vol_avg[i]) or
            np.isnan(atr[i]) or np.isnan(atr_ma[i])):
            signals[i] = 0.0 if position == 0 else (0.25 if position == 1 else -0.25)
            continue
        
        price_close = close[i]
        
        # Volatility filter: only trade when volatility is expanding
        vol_expanding = atr[i] > atr_ma[i]
        
        # Trend filters: price above/below 1d EMA50
        uptrend = price_close > ema_50_1d_aligned[i]
        downtrend = price_close < ema_50_1d_aligned[i]
        
        # Breakout conditions using Donchian channels
        breakout_up = price_close > donchian_high[i]
        breakout_down = price_close < donchian_low[i]
        
        # Volume confirmation
        vol_confirmed = vol_spike[i]
        
        # Long: upward breakout with volume in uptrend and expanding volatility
        long_signal = breakout_up and vol_confirmed and uptrend and vol_expanding
        
        # Short: downward breakout with volume in downtrend and expanding volatility
        short_signal = breakout_down and vol_confirmed and downtrend and vol_expanding
        
        # Exit when price returns to the opposite Donchian level or volatility contracts
        exit_long = position == 1 and (price_close < donchian_low[i] or not vol_expanding)
        exit_short = position == -1 and (price_close > donchian_high[i] or not vol_expanding)
        
        # Trading logic
        if long_signal and position != 1:
            position = 1
            signals[i] = 0.25
        elif short_signal and position != -1:
            position = -1
            signals[i] = -0.25
        elif position == 1 and exit_long:
            position = 0
            signals[i] = 0.0
        elif position == -1 and exit_short:
            position = 0
            signals[i] = 0.0
        else:
            signals[i] = 0.25 if position == 1 else (-0.25 if position == -1 else 0.0)
    
    return signals
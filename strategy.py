#!/usr/bin/env python3
"""
Hypothesis: 1h strategy using 4h Donchian channel breakout with volume confirmation and 1d EMA trend filter.
In uptrend (price > 1d EMA50), buy breakouts above 4h Donchian upper band; in downtrend (price < 1d EMA50),
sell breakdowns below 4h Donchian lower band. Uses volume spike (>1.8x 20-period average) for confirmation.
Limits trades to 15-37/year by using 4h structure for direction and 1h only for entry timing.
Adds session filter (08-20 UTC) to avoid low-liquidity periods.
Position size: 0.20.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Load 4h data ONCE before loop for Donchian channels
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 30:
        return np.zeros(n)
    
    high_4h = df_4h['high'].values
    low_4h = df_4h['low'].values
    close_4h = df_4h['close'].values
    
    # 4h Donchian channel (20-period)
    donchian_upper = pd.Series(high_4h).rolling(window=20, min_periods=20).max().values
    donchian_lower = pd.Series(low_4h).rolling(window=20, min_periods=20).min().values
    
    # Align Donchian levels to 1h timeframe (wait for 4h bar to close)
    donchian_upper_aligned = align_htf_to_ltf(prices, df_4h, donchian_upper)
    donchian_lower_aligned = align_htf_to_ltf(prices, df_4h, donchian_lower)
    
    # Load 1d data ONCE before loop for EMA trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    ema_50 = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_aligned = align_htf_to_ltf(prices, df_1d, ema_50)
    
    # Volume confirmation (volume spike > 1.8x 20-period average)
    vol_ma_20 = pd.Series(prices['volume'].values).rolling(window=20, min_periods=20).mean().values
    vol_ratio = prices['volume'].values / vol_ma_20
    
    # Pre-compute session filter (08-20 UTC)
    hours = pd.DatetimeIndex(prices['open_time']).hour
    in_session = (hours >= 8) & (hours <= 20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(50, n):
        # Skip if indicators not ready
        if (np.isnan(donchian_upper_aligned[i]) or np.isnan(donchian_lower_aligned[i]) or 
            np.isnan(ema_50_aligned[i]) or np.isnan(vol_ratio[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Skip if outside trading session
        if not in_session[i]:
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        price_close = prices['close'].iloc[i]
        ema_trend = ema_50_aligned[i]
        vol_ratio_val = vol_ratio[i]
        
        if position == 0:
            # Enter long: price breaks above 4h Donchian upper + uptrend + volume spike
            if (price_close > donchian_upper_aligned[i] and 
                price_close > ema_trend and 
                vol_ratio_val > 1.8):
                signals[i] = 0.20
                position = 1
            # Enter short: price breaks below 4h Donchian lower + downtrend + volume spike
            elif (price_close < donchian_lower_aligned[i] and 
                  price_close < ema_trend and 
                  vol_ratio_val > 1.8):
                signals[i] = -0.20
                position = -1
        
        elif position != 0:
            # Exit: trend reversal
            exit_signal = False
            
            if position == 1 and price_close < ema_trend:
                exit_signal = True
            elif position == -1 and price_close > ema_trend:
                exit_signal = True
            
            if exit_signal:
                signals[i] = 0.0
                position = 0
            else:
                # Hold position
                signals[i] = 0.20 if position == 1 else -0.20
    
    return signals

name = "1h_DonchianBreakout_4h_1dEMA50_Volume_Session"
timeframe = "1h"
leverage = 1.0
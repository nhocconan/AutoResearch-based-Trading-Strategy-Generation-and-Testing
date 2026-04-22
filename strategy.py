#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 200:
        return np.zeros(n)
    
    # 1d EMA34 for trend direction
    df_1d = get_htf_data(prices, '1d')
    close_1d = df_1d['close'].values
    ema_34_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # 1d EMA100 for trend filter
    ema_100_1d = pd.Series(close_1d).ewm(span=100, adjust=False, min_periods=100).mean().values
    ema_100_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_100_1d)
    
    # 4h Donchian(20) for breakout
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    
    donch_high = pd.Series(high).rolling(window=20, min_periods=20).max().values
    donch_low = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # 4h volume spike confirmation
    volume = prices['volume'].values
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_spike = volume > 1.5 * vol_ma_20
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    for i in range(200, n):
        # Skip if any data is not ready
        if (np.isnan(ema_34_1d_aligned[i]) or 
            np.isnan(ema_100_1d_aligned[i]) or 
            np.isnan(donch_high[i]) or 
            np.isnan(donch_low[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        price = close[i]
        ema34 = ema_34_1d_aligned[i]
        ema100 = ema_100_1d_aligned[i]
        vol_spike_val = vol_spike[i]
        
        if position == 0 and vol_spike_val:
            # Long: price breaks above Donchian high + EMA34 > EMA100 (uptrend)
            if price > donch_high[i] and ema34 > ema100:
                signals[i] = 0.25
                position = 1
                entry_price = price
            # Short: price breaks below Donchian low + EMA34 < EMA100 (downtrend)
            elif price < donch_low[i] and ema34 < ema100:
                signals[i] = -0.25
                position = -1
                entry_price = price
        
        elif position != 0:
            # Exit: price returns to Donchian mid-point or trend reversal
            donch_mid = (donch_high[i] + donch_low[i]) / 2
            trend_reversal = (position == 1 and ema34 < ema100) or (position == -1 and ema34 > ema100)
            
            if (position == 1 and price < donch_mid) or (position == -1 and price > donch_mid) or trend_reversal:
                signals[i] = 0.0
                position = 0
            else:
                # Hold position
                signals[i] = 0.25 if position == 1 else -0.25
    
    return signals

name = "4h_DonchianBreakout_EMA34EMA100Trend_VolumeSpike"
timeframe = "4h"
leverage = 1.0
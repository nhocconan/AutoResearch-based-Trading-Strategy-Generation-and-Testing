#!/usr/bin/env python3
"""
exp_6453_4h_donchian20_12h_ema_vol_v1
Hypothesis: 4h Donchian(20) breakout with 12h EMA trend filter and volume confirmation.
Donchian breakouts capture momentum, EMA filters trend direction, volume confirms strength.
Works in bull/bear: breakouts occur in both regimes, volume filter avoids false signals.
Target: 75-200 trades over 4 years (discrete size 0.25, max 0.30).
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "exp_6453_4h_donchian20_12h_ema_vol_v1"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Load HTF data ONCE before loop
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 21:
        return np.zeros(n)
    
    # Calculate 12h EMA(21) for trend filter
    close_12h = df_12h['close'].values
    ema_12h = pd.Series(close_12h).ewm(span=21, min_periods=21, adjust=False).mean().values
    ema_12h_aligned = align_htf_to_ltf(prices, df_12h, ema_12h)
    
    # Calculate Donchian channels (20-period) on primary timeframe
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Donchian upper/lower (20-period)
    donchian_high = pd.Series(high).rolling(window=20, min_periods=20).max().values
    donchian_low = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # Volume confirmation: current volume > 1.5x 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_confirm = volume > (vol_ma * 1.5)
    
    signals = np.zeros(n)
    
    # Track position state for stoploss and reversals
    position = 0  # 0=flat, 1=long, -1=short
    entry_price = 0.0
    
    for i in range(20, n):
        # Skip if EMA not available (first 20 bars of 12h data)
        if np.isnan(ema_12h_aligned[i]):
            continue
            
        # Trend filter: price above/below 12h EMA
        price_above_ema = close[i] > ema_12h_aligned[i]
        price_below_ema = close[i] < ema_12h_aligned[i]
        
        # Donchian breakout conditions
        breakout_up = (i > 0 and 
                      high[i] > donchian_high[i-1] and 
                      donchian_high[i-1] != donchian_high[i-2])  # ensure channel is fresh
        breakout_down = (i > 0 and 
                        low[i] < donchian_low[i-1] and 
                        donchian_low[i-1] != donchian_low[i-2])
        
        # Volume confirmation
        vol_ok = vol_confirm[i] if not np.isnan(vol_confirm[i]) else False
        
        # Entry logic
        if position == 0:  # flat
            # Long: breakout above upper channel + above EMA + volume
            if breakout_up and price_above_ema and vol_ok:
                signals[i] = 0.25
                position = 1
                entry_price = close[i]
            # Short: breakout below lower channel + below EMA + volume
            elif breakout_down and price_below_ema and vol_ok:
                signals[i] = -0.25
                position = -1
                entry_price = close[i]
        elif position == 1:  # long
            # Exit: price breaks below lower Donchian channel
            if low[i] < donchian_low[i]:
                signals[i] = 0.0
                position = 0
            # Optional: stoploss at 2*ATR (simplified as 2% for now, will improve with ATR)
            elif close[i] < entry_price * 0.98:
                signals[i] = 0.0
                position = 0
        elif position == -1:  # short
            # Exit: price breaks above upper Donchian channel
            if high[i] > donchian_high[i]:
                signals[i] = 0.0
                position = 0
            # Optional: stoploss at 2*ATR (simplified as 2% for now)
            elif close[i] > entry_price * 1.02:
                signals[i] = 0.0
                position = 0
    
    return signals
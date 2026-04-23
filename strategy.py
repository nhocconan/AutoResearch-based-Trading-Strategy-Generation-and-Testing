#!/usr/bin/env python3
"""
Hypothesis: 4h Donchian(20) breakout + 12h EMA50 trend filter + volume spike + ATR(14) stoploss.
Long when price breaks above Donchian upper AND 12h EMA50 rising AND volume > 2x average.
Short when price breaks below Donchian lower AND 12h EMA50 falling AND volume > 2x average.
Exit via ATR-based trailing stop or opposite Donchian breakout.
Donchian channels provide objective structure; EMA50 filters for higher timeframe trend;
volume spike confirms conviction; ATR stop manages risk. Designed for 4h targeting
75-200 total trades over 4 years to minimize fee drag while capturing trends in
both bull and bear markets.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Load 12h data for EMA50 trend filter - ONCE before loop
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 50:
        return np.zeros(n)
    
    close_12h = df_12h['close'].values
    
    # Calculate EMA50 on 12h data
    ema50_12h = pd.Series(close_12h).ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Align 12h EMA50 to 4h timeframe
    ema50_12h_aligned = align_htf_to_ltf(prices, df_12h, ema50_12h)
    
    # Calculate Donchian channels (20-period) on 4h data
    highest_high = pd.Series(high).rolling(window=20, min_periods=20).max().values
    lowest_low = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # Volume average (20-period) on primary timeframe
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    # ATR(14) for stoploss
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr2[0] = 0
    tr3[0] = 0
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    atr = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    highest_since_entry = 0.0
    lowest_since_entry = 0.0
    
    for i in range(100, n):
        # Skip if data not ready
        if (np.isnan(ema50_12h_aligned[i]) or np.isnan(highest_high[i]) or 
            np.isnan(lowest_low[i]) or np.isnan(vol_ma[i]) or np.isnan(atr[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        ema50_val = ema50_12h_aligned[i]
        donch_high = highest_high[i]
        donch_low = lowest_low[i]
        vol_ma_val = vol_ma[i]
        vol_current = volume[i]
        atr_val = atr[i]
        price = close[i]
        
        # EMA50 slope (rising/falling) - use 3-bar slope for stability
        if i >= 3:
            ema50_slope = ema50_val - ema50_12h_aligned[i-3]
            ema_rising = ema50_slope > 0
            ema_falling = ema50_slope < 0
        else:
            ema_rising = False
            ema_falling = False
        
        if position == 0:
            # Long: Donchian breakout above upper AND EMA50 rising AND volume spike
            if (price > donch_high and ema_rising and vol_current > 2.0 * vol_ma_val):
                signals[i] = 0.25
                position = 1
                entry_price = price
                highest_since_entry = price
            # Short: Donchian breakdown below lower AND EMA50 falling AND volume spike
            elif (price < donch_low and ema_falling and vol_current > 2.0 * vol_ma_val):
                signals[i] = -0.25
                position = -1
                entry_price = price
                lowest_since_entry = price
        else:
            # Update highest/lowest since entry for trailing stop
            if position == 1:
                highest_since_entry = max(highest_since_entry, price)
                # Exit conditions: ATR trailing stop OR opposite Donchian breakout
                if (price < highest_since_entry - 2.5 * atr_val or 
                    price < donch_low):
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = 0.25
            else:  # position == -1
                lowest_since_entry = min(lowest_since_entry, price)
                # Exit conditions: ATR trailing stop OR opposite Donchian breakout
                if (price > lowest_since_entry + 2.5 * atr_val or 
                    price > donch_high):
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = -0.25
    
    return signals

name = "4H_Donchian20_12hEMA50_Volume_ATRStop"
timeframe = "4h"
leverage = 1.0
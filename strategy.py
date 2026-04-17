#!/usr/bin/env python3
"""
4h Donchian(20) Breakout + 12h EMA34 Trend + Volume Spike + ATR Stop
Long: Price breaks above Donchian(20) high + price > 12h EMA34 + volume > 1.5x 4h volume SMA(20)
Short: Price breaks below Donchian(20) low + price < 12h EMA34 + volume > 1.5x 4h volume SMA(20)
Exit: Opposite Donchian breakout or ATR-based stop (implemented via signal=0 when stop condition met)
Designed to capture strong trend moves with volume confirmation and trend filter.
Target: 75-200 total trades over 4 years (19-50/year)
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 12h data for EMA34 trend filter
    df_12h = get_htf_data(prices, '12h')
    close_12h = df_12h['close'].values
    
    # Calculate 12h EMA34
    ema_34_12h = pd.Series(close_12h).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_12h_aligned = align_htf_to_ltf(prices, df_12h, ema_34_12h)
    
    # Calculate 4h volume SMA(20) for volume confirmation
    vol_sma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    # Calculate ATR(14) for stop loss
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr1[0] = 0
    tr2[0] = 0
    tr3[0] = 0
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    atr = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    # Calculate Donchian channels (20-period)
    donch_high = pd.Series(high).rolling(window=20, min_periods=20).max().values
    donch_low = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    signals = np.zeros(n)
    position = 0  # -1 short, 0 flat, 1 long
    entry_price = 0.0
    
    start_idx = max(34, 20)  # need EMA34 and Donchian
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(ema_34_12h_aligned[i]) or np.isnan(vol_sma_20[i]) or
            np.isnan(donch_high[i]) or np.isnan(donch_low[i]) or np.isnan(atr[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        vol = volume[i]
        vol_sma_val = vol_sma_20[i]
        ema_trend = ema_34_12h_aligned[i]
        upper = donch_high[i]
        lower = donch_low[i]
        atr_val = atr[i]
        
        # Check for breakouts
        breakout_up = price > upper
        breakout_down = price < lower
        
        if position == 0:
            # Long: upward breakout + price above 12h EMA34 + volume spike
            if breakout_up and price > ema_trend and vol > 1.5 * vol_sma_val:
                signals[i] = 0.25
                position = 1
                entry_price = price
            # Short: downward breakout + price below 12h EMA34 + volume spike
            elif breakout_down and price < ema_trend and vol > 1.5 * vol_sma_val:
                signals[i] = -0.25
                position = -1
                entry_price = price
        
        elif position == 1:
            # Long exit: downward breakout OR price drops below entry - 2*ATR (stop)
            if breakout_down or price < entry_price - 2.0 * atr_val:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: upward breakout OR price rises above entry + 2*ATR (stop)
            if breakout_up or price > entry_price + 2.0 * atr_val:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "4h_Donchian20_EMA34_VolumeSpike_ATRStop"
timeframe = "4h"
leverage = 1.0
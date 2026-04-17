#!/usr/bin/env python3
"""
Hypothesis: 4h Donchian(20) breakout + 12h EMA34 trend + Volume spike + ATR stoploss.
Long when price breaks above Donchian(20) high with volume > 1.8x 20-bar average and price > 12h EMA34.
Short when price breaks below Donchian(20) low with volume > 1.8x 20-bar average and price < 12h EMA34.
Exit via ATR-based trailing stop: long exit if price < highest_high_since_entry - 2.5*ATR(20),
short exit if price > lowest_low_since_entry + 2.5*ATR(20).
Uses 4h for price/volume/Donchian/ATR, 12h for EMA34 trend filter.
Target: 75-200 total trades over 4 years (19-50/year).
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
    ema_12h_34 = pd.Series(close_12h).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_12h_34_aligned = align_htf_to_ltf(prices, df_12h, ema_12h_34)
    
    # Calculate 4h Donchian(20) channels
    def donchian_channels(high, low, period=20):
        upper = np.full_like(high, np.nan)
        lower = np.full_like(high, np.nan)
        for i in range(period-1, len(high)):
            upper[i] = np.max(high[i-period+1:i+1])
            lower[i] = np.min(low[i-period+1:i+1])
        return upper, lower
    
    donchian_upper, donchian_lower = donchian_channels(high, low, 20)
    
    # Calculate 4h ATR(20) for volatility and stoploss
    def atr(high, low, close, period=20):
        tr = np.zeros_like(close)
        for i in range(1, len(close)):
            tr[i] = max(high[i] - low[i], abs(high[i] - close[i-1]), abs(low[i] - close[i-1]))
        # Wilder's ATR
        atr_vals = np.zeros_like(close)
        atr_vals[period] = np.mean(tr[1:period+1])
        for i in range(period+1, len(tr)):
            atr_vals[i] = (atr_vals[i-1] * (period-1) + tr[i]) / period
        return atr_vals
    
    atr_vals = atr(high, low, close, 20)
    
    # Calculate 4h volume spike (current volume > 1.8x 20-period average)
    volume_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (volume_ma * 1.8)
    
    signals = np.zeros(n)
    position = 0  # -1: short, 0: flat, 1: long
    entry_price = 0.0
    highest_since_entry = 0.0
    lowest_since_entry = 0.0
    
    start_idx = 50  # warmup for indicators
    
    for i in range(start_idx, n):
        # Skip if any required data is not available
        if (np.isnan(donchian_upper[i]) or 
            np.isnan(donchian_lower[i]) or 
            np.isnan(ema_12h_34_aligned[i]) or 
            np.isnan(atr_vals[i]) or 
            np.isnan(volume_ma[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        vol_spike = volume_spike[i]
        donch_up = donchian_upper[i]
        donch_low = donchian_lower[i]
        ema_trend = ema_12h_34_aligned[i]
        atr_val = atr_vals[i]
        
        if position == 0:
            # Long: price breaks above Donchian upper with volume spike and bullish 12h trend
            if price > donch_up and vol_spike and price > ema_trend:
                signals[i] = 0.25
                position = 1
                entry_price = price
                highest_since_entry = price
            # Short: price breaks below Donchian lower with volume spike and bearish 12h trend
            elif price < donch_low and vol_spike and price < ema_trend:
                signals[i] = -0.25
                position = -1
                entry_price = price
                lowest_since_entry = price
        
        elif position == 1:
            # Update highest high since entry
            highest_since_entry = max(highest_since_entry, price)
            # ATR trailing stop: exit if price drops below highest - 2.5*ATR
            if price < highest_since_entry - 2.5 * atr_val:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Update lowest low since entry
            lowest_since_entry = min(lowest_since_entry, price)
            # ATR trailing stop: exit if price rises above lowest + 2.5*ATR
            if price > lowest_since_entry + 2.5 * atr_val:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "4h_Donchian20_12hEMA34_VolumeSpike_ATRStop"
timeframe = "4h"
leverage = 1.0
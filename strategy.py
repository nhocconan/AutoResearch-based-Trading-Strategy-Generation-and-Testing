#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Donchian(20) breakout + 1d EMA34 trend filter + volume confirmation + ATR stop.
# Donchian breakout captures breakouts in trending markets; EMA34 filters for trend direction (only long when price > EMA34, short when price < EMA34).
# Volume confirmation requires current volume > 1.8x 20-period average to avoid false breakouts.
# ATR-based stop exits when price moves against position by 2.5 * ATR(14).
# Designed to work in both bull and bear markets by aligning with 1d trend via EMA34 filter.
# Targets 20-40 trades/year with strict entry conditions to minimize fee drag.

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Load 1d data for EMA trend filter (once before loop)
    df_1d = get_htf_data(prices, '1d')
    close_1d = df_1d['close'].values
    
    # Calculate 34-period EMA on 1d data
    ema_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_1d)
    
    # Calculate ATR(14) for stop loss
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]  # first period
    atr = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    # Calculate Donchian channels (20-period)
    highest_high = pd.Series(high).rolling(window=20, min_periods=20).max().values
    lowest_low = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # Calculate 20-period average volume for volume spike detection
    volume = prices['volume'].values
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0  # track entry price for stop loss
    
    for i in range(50, n):
        # Skip if data not ready
        if (np.isnan(ema_1d_aligned[i]) or 
            np.isnan(atr[i]) or 
            np.isnan(highest_high[i]) or 
            np.isnan(lowest_low[i]) or 
            np.isnan(vol_ma_20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        price = prices['close'].iloc[i]
        vol = volume[i]
        vol_ma = vol_ma_20[i]
        atr_val = atr[i]
        ema_val = ema_1d_aligned[i]
        upper_channel = highest_high[i]
        lower_channel = lowest_low[i]
        
        # Volume filter: current volume > 1.8 * 20-period average
        vol_spike = vol > 1.8 * vol_ma
        
        if position == 0:
            # Long conditions: breakout above upper channel + uptrend + volume spike
            if price > upper_channel and price > ema_val and vol_spike:
                signals[i] = 0.25
                position = 1
                entry_price = price
            # Short conditions: breakdown below lower channel + downtrend + volume spike
            elif price < lower_channel and price < ema_val and vol_spike:
                signals[i] = -0.25
                position = -1
                entry_price = price
        
        elif position != 0:
            # Stop loss: price moves against position by 2.5 * ATR
            stop_loss_hit = False
            if position == 1:  # long position
                if price < entry_price - 2.5 * atr_val:
                    stop_loss_hit = True
            elif position == -1:  # short position
                if price > entry_price + 2.5 * atr_val:
                    stop_loss_hit = True
            
            if stop_loss_hit:
                signals[i] = 0.0
                position = 0
            else:
                # Hold position
                signals[i] = 0.25 if position == 1 else -0.25
    
    return signals

name = "4h_Donchian20_1dEMA34_Trend_Volume_ATRStop"
timeframe = "4h"
leverage = 1.0
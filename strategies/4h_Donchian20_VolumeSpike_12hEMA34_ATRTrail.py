#!/usr/bin/env python3
"""
Hypothesis: 4h Donchian(20) breakout with volume confirmation, 12h EMA34 trend filter, and ATR(10) trailing stop.
- Long when price breaks above Donchian(20) high + volume > 2.0x 20-period 4h volume MA + price above 12h EMA34
- Short when price breaks below Donchian(20) low + volume > 2.0x 20-period 4h volume MA + price below 12h EMA34
- Fixed position size 0.30 to balance return and drawdown
- ATR-based trailing stop (1.5x ATR) to lock in profits
- Designed for moderate trade frequency (target: 100-200 trades over 4 years) to avoid fee drag
- Works in bull markets (buying breakouts with uptrend) and bear markets (selling breakdowns with downtrend)
- Uses proven BTC/ETH edge: Donchian breakouts + volume spike + HTF trend filter
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
    
    # Get 12h data for EMA34 trend filter (HTF)
    df_12h = get_htf_data(prices, '12h')
    close_12h = df_12h['close'].values
    
    # Calculate 12h EMA34
    ema_34_12h = pd.Series(close_12h).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_12h_aligned = align_htf_to_ltf(prices, df_12h, ema_34_12h)
    
    # Get 4h data for Donchian channels and volume confirmation (HTF for structure)
    df_4h = get_htf_data(prices, '4h')
    high_4h = df_4h['high'].values
    low_4h = df_4h['low'].values
    close_4h = df_4h['close'].values
    volume_4h = df_4h['volume'].values
    
    # Donchian channels (20-period) on 4h
    donch_high_20 = pd.Series(high_4h).rolling(window=20, min_periods=20).max().values
    donch_low_20 = pd.Series(low_4h).rolling(window=20, min_periods=20).min().values
    
    # Volume average (20-period) on 4h for confirmation
    volume_ma_20_4h = pd.Series(volume_4h).rolling(window=20, min_periods=20).mean().values
    
    # ATR (10-period) on 4h for stoploss
    tr1 = high_4h - low_4h
    tr2 = np.abs(high_4h - np.roll(close_4h, 1))
    tr3 = np.abs(low_4h - np.roll(close_4h, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]  # first period
    atr_10_4h = pd.Series(tr).rolling(window=10, min_periods=10).mean().values
    
    # Align all 4h indicators to primary timeframe (4h is primary, so no alignment needed)
    # But we still call align_htf_to_ltf for safety and consistency with rules
    donch_high_aligned = align_htf_to_ltf(prices, df_4h, donch_high_20)
    donch_low_aligned = align_htf_to_ltf(prices, df_4h, donch_low_20)
    volume_ma_aligned = align_htf_to_ltf(prices, df_4h, volume_ma_20_4h)
    atr_aligned = align_htf_to_ltf(prices, df_4h, atr_10_4h)
    
    # Align 12h EMA34 to primary timeframe
    ema_34_aligned = align_htf_to_ltf(prices, df_12h, ema_34_12h)
    
    signals = np.zeros(n)
    position = 0  # -1: short, 0: flat, 1: long
    entry_price = 0.0
    atr_stop = 0.0
    
    start_idx = 100  # warmup
    
    for i in range(start_idx, n):
        if (np.isnan(donch_high_aligned[i]) or np.isnan(donch_low_aligned[i]) or 
            np.isnan(volume_ma_aligned[i]) or np.isnan(atr_aligned[i]) or 
            np.isnan(ema_34_aligned[i])):
            signals[i] = 0.0
            continue
        
        donch_high = donch_high_aligned[i]
        donch_low = donch_low_aligned[i]
        vol_ma = volume_ma_aligned[i]
        atr_val = atr_aligned[i]
        ema_34_val = ema_34_aligned[i]
        vol = volume[i]
        price = close[i]
        
        if position == 0:
            # Look for breakouts with volume confirmation and 12h EMA34 trend filter
            # Long: price breaks above Donchian(20) high + volume spike + price above 12h EMA34
            if price > donch_high and vol > 2.0 * vol_ma and price > ema_34_val:
                signals[i] = 0.30
                position = 1
                entry_price = price
                atr_stop = entry_price - 1.5 * atr_val
            # Short: price breaks below Donchian(20) low + volume spike + price below 12h EMA34
            elif price < donch_low and vol > 2.0 * vol_ma and price < ema_34_val:
                signals[i] = -0.30
                position = -1
                entry_price = price
                atr_stop = entry_price + 1.5 * atr_val
        
        elif position == 1:
            # Check stoploss
            if price <= atr_stop:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.30
                # Trail stop: raise stop if price moves favorably
                atr_stop = max(atr_stop, price - 1.0 * atr_val)
        
        elif position == -1:
            # Check stoploss
            if price >= atr_stop:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.30
                # Trail stop: lower stop if price moves favorably
                atr_stop = min(atr_stop, price + 1.0 * atr_val)
    
    return signals

name = "4h_Donchian20_VolumeSpike_12hEMA34_ATRTrail"
timeframe = "4h"
leverage = 1.0
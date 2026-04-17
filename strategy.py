#!/usr/bin/env python3
"""
Hypothesis: 6h Camarilla R3/S3 breakout with volume confirmation and 12h EMA34 trend filter.
- Long when price closes above Camarilla R3 (6h) + volume > 1.5x 20-period 6h volume MA + price above 12h EMA34
- Short when price closes below Camarilla S3 (6h) + volume > 1.5x 20-period 6h volume MA + price below 12h EMA34
- Fixed position size 0.25 to limit fee churn and manage drawdown
- ATR-based trailing stop (2.0x ATR) to lock in profits
- Designed for low trade frequency (target: 50-150 trades over 4 years) to avoid fee drag
- Works in bull markets (buying breakouts above 12h EMA34) and bear markets (selling breakdowns below 12h EMA34)
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
    
    # Get 6h data for Camarilla levels, volume confirmation, and ATR (primary timeframe)
    df_6h = get_htf_data(prices, '6h')
    high_6h = df_6h['high'].values
    low_6h = df_6h['low'].values
    close_6h = df_6h['close'].values
    volume_6h = df_6h['volume'].values
    
    # Calculate Camarilla levels for 6h (based on previous 6h bar)
    # Camarilla: R3 = C + (H-L)*1.1/4, S3 = C - (H-L)*1.1/4
    # We use the previous completed 6h bar's OHLC
    prev_close_6h = np.roll(close_6h, 1)
    prev_high_6h = np.roll(high_6h, 1)
    prev_low_6h = np.roll(low_6h, 1)
    prev_close_6h[0] = close_6h[0]  # first period
    prev_high_6h[0] = high_6h[0]
    prev_low_6h[0] = low_6h[0]
    
    camarilla_r3 = prev_close_6h + (prev_high_6h - prev_low_6h) * 1.1 / 4
    camarilla_s3 = prev_close_6h - (prev_high_6h - prev_low_6h) * 1.1 / 4
    
    # Volume average (20-period) on 6h for confirmation
    volume_ma_20 = pd.Series(volume_6h).rolling(window=20, min_periods=20).mean().values
    
    # ATR (10-period) on 6h for stoploss
    tr1 = high_6h - low_6h
    tr2 = np.abs(high_6h - np.roll(close_6h, 1))
    tr3 = np.abs(low_6h - np.roll(close_6h, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]  # first period
    atr_10 = pd.Series(tr).rolling(window=10, min_periods=10).mean().values
    
    # Get 12h data for EMA34 trend filter (HTF)
    df_12h = get_htf_data(prices, '12h')
    close_12h = df_12h['close'].values
    
    # Calculate 12h EMA34
    ema_34_12h = pd.Series(close_12h).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_12h_aligned = align_htf_to_ltf(prices, df_12h, ema_34_12h)
    
    # Align all indicators to 6h timeframe (primary)
    r3_aligned = align_htf_to_ltf(prices, df_6h, camarilla_r3)
    s3_aligned = align_htf_to_ltf(prices, df_6h, camarilla_s3)
    volume_ma_aligned = align_htf_to_ltf(prices, df_6h, volume_ma_20)
    atr_aligned = align_htf_to_ltf(prices, df_6h, atr_10)
    
    signals = np.zeros(n)
    position = 0  # -1: short, 0: flat, 1: long
    entry_price = 0.0
    atr_stop = 0.0
    
    start_idx = 100  # warmup
    
    for i in range(start_idx, n):
        if (np.isnan(r3_aligned[i]) or np.isnan(s3_aligned[i]) or 
            np.isnan(volume_ma_aligned[i]) or np.isnan(atr_aligned[i]) or 
            np.isnan(ema_34_12h_aligned[i])):
            signals[i] = 0.0
            continue
        
        r3_val = r3_aligned[i]
        s3_val = s3_aligned[i]
        vol_ma = volume_ma_aligned[i]
        atr_val = atr_aligned[i]
        ema_34_val = ema_34_12h_aligned[i]
        vol = volume[i]
        price = close[i]
        
        if position == 0:
            # Look for breakouts with volume confirmation and 12h EMA34 trend filter
            # Long: price closes above R3 + volume spike + price above 12h EMA34
            if price > r3_val and vol > 1.5 * vol_ma and price > ema_34_val:
                signals[i] = 0.25
                position = 1
                entry_price = price
                atr_stop = entry_price - 2.0 * atr_val
            # Short: price closes below S3 + volume spike + price below 12h EMA34
            elif price < s3_val and vol > 1.5 * vol_ma and price < ema_34_val:
                signals[i] = -0.25
                position = -1
                entry_price = price
                atr_stop = entry_price + 2.0 * atr_val
        
        elif position == 1:
            # Check stoploss
            if price <= atr_stop:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
                # Trail stop: raise stop if price moves favorably
                atr_stop = max(atr_stop, price - 1.5 * atr_val)
        
        elif position == -1:
            # Check stoploss
            if price >= atr_stop:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
                # Trail stop: lower stop if price moves favorably
                atr_stop = min(atr_stop, price + 1.5 * atr_val)
    
    return signals

name = "6h_Camarilla_R3S3_12hEMA34_VolumeSpike_ATRTrail"
timeframe = "6h"
leverage = 1.0
#!/usr/bin/env python3
"""
Hypothesis: 6h strategy using 12h Camarilla R3/S3 levels with 1d EMA34 trend filter and volume confirmation
- Uses 1d EMA34 slope for multi-timeframe trend bias (long when rising, short when falling)
- Breakout triggers when price closes beyond R3 (long) or S3 (short) with volume > 1.8x 20-period MA
- Fixed position size 0.25 to limit fee churn and manage drawdown
- ATR-based trailing stop (2.0x ATR) to lock in profits and reduce losses
- Designed to work in bull markets (buying R3 breakouts in uptrends) and bear markets (selling S3 breakdowns in downtrends)
- Weekly pivot regime filter: only trade in alignment with weekly trend (above/below weekly pivot)
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
    
    # Get 1d data for EMA34 trend filter (HTF)
    df_1d = get_htf_data(prices, '1d')
    close_1d = df_1d['close'].values
    
    # Calculate 1d EMA34 and its slope
    ema34_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema34_slope = np.gradient(ema34_1d)  # slope of EMA34
    
    # Get 12h data for Camarilla pivot levels (HTF)
    df_12h = get_htf_data(prices, '12h')
    high_12h = df_12h['high'].values
    low_12h = df_12h['low'].values
    close_12h = df_12h['close'].values
    
    # Calculate Camarilla levels (R3, S3) from previous completed 12h bar
    rng = high_12h - low_12h
    r3 = close_12h + 1.1 * rng / 2
    s3 = close_12h - 1.1 * rng / 2
    # Shift by 1 to use only completed 12h bars (avoid look-ahead)
    r3_prev = np.roll(r3, 1)
    s3_prev = np.roll(s3, 1)
    r3_prev[0] = r3[0]
    s3_prev[0] = s3[0]
    
    # Get weekly data for regime filter (weekly pivot)
    df_1w = get_htf_data(prices, '1w')
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    
    # Calculate weekly pivot point (PP) from previous completed 1w bar
    pp_1w = (high_1w + low_1w + close_1w) / 3.0
    # Shift by 1 to use only completed weekly bars
    pp_1w_prev = np.roll(pp_1w, 1)
    pp_1w_prev[0] = pp_1w[0]
    
    # Get 6h data for volume confirmation and ATR (primary timeframe)
    df_6h = get_htf_data(prices, '6h')
    volume_6h = df_6h['volume'].values
    high_6h = df_6h['high'].values
    low_6h = df_6h['low'].values
    close_6h = df_6h['close'].values
    
    # Volume average (20-period) on 6h
    volume_ma_20 = pd.Series(volume_6h).rolling(window=20, min_periods=20).mean().values
    
    # ATR (14-period) on 6h for stoploss
    tr1 = high_6h - low_6h
    tr2 = np.abs(high_6h - np.roll(close_6h, 1))
    tr3 = np.abs(low_6h - np.roll(close_6h, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]  # first period
    atr_14 = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    # Align all indicators to 6h timeframe (primary)
    r3_aligned = align_htf_to_ltf(prices, df_12h, r3_prev)
    s3_aligned = align_htf_to_ltf(prices, df_12h, s3_prev)
    ema34_slope_aligned = align_htf_to_ltf(prices, df_1d, ema34_slope)
    pp_1w_aligned = align_htf_to_ltf(prices, df_1w, pp_1w_prev)
    volume_ma_aligned = align_htf_to_ltf(prices, df_6h, volume_ma_20)
    atr_aligned = align_htf_to_ltf(prices, df_6h, atr_14)
    
    signals = np.zeros(n)
    position = 0  # -1: short, 0: flat, 1: long
    entry_price = 0.0
    atr_stop = 0.0
    
    start_idx = 100  # warmup
    
    for i in range(start_idx, n):
        if (np.isnan(r3_aligned[i]) or np.isnan(s3_aligned[i]) or 
            np.isnan(ema34_slope_aligned[i]) or np.isnan(pp_1w_aligned[i]) or 
            np.isnan(volume_ma_aligned[i]) or np.isnan(atr_aligned[i])):
            signals[i] = 0.0
            continue
        
        r3_val = r3_aligned[i]
        s3_val = s3_aligned[i]
        ema_slope = ema34_slope_aligned[i]
        pp_1w = pp_1w_aligned[i]
        vol_ma = volume_ma_aligned[i]
        atr_val = atr_aligned[i]
        vol = volume[i]
        price = close[i]
        
        if position == 0:
            # Look for breakouts with volume confirmation and trend filter
            # Weekly regime: only long above weekly pivot, short below weekly pivot
            # Long: price closes above R3 + volume spike + EMA34 rising + price > weekly PP
            if price > r3_val and vol > 1.8 * vol_ma and ema_slope > 0 and price > pp_1w:
                signals[i] = 0.25
                position = 1
                entry_price = price
                atr_stop = entry_price - 2.0 * atr_val
            # Short: price closes below S3 + volume spike + EMA34 falling + price < weekly PP
            elif price < s3_val and vol > 1.8 * vol_ma and ema_slope < 0 and price < pp_1w:
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

name = "6h_Camarilla_R3S3_1dEMA34_VolumeSpike_ATRTrail_WeeklyPP"
timeframe = "6h"
leverage = 1.0
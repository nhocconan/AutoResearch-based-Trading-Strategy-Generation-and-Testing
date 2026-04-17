#!/usr/bin/env python3
"""
Hypothesis: 4h Camarilla pivot (H3/L3) breakout + 1d EMA34 trend filter + volume spike + ATR trailing stop
- Uses 1d Camarilla levels (H3/L3) calculated from previous day's range for institutional reference points
- Trend filter: 1d EMA34 slope confirms bias (long only when EMA34 rising, short only when falling)
- Volume confirmation: 1.5x 20-period MA on 4h filters weak breakouts
- ATR trailing stop (2.0x ATR) adapts to volatility and reduces drawdown
- Fixed position size 0.25 balances risk and avoids fee churn from dynamic sizing changes
- Works in bull markets (buying H3 breakouts in uptrend) and bear markets (selling L3 breakdowns in downtrend)
- Proven pattern: Camarilla breaks with volume show strong test performance in DB
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
    
    # Get 1d data for Camarilla pivot and EMA34 trend (HTF)
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate Camarilla levels (H3, L3) from previous completed 1d bar
    # Camarilla: H3 = close + 1.1*(high-low)/4, L3 = close - 1.1*(high-low)/4
    rng = high_1d - low_1d
    h3 = close_1d + 1.1 * rng / 4
    l3 = close_1d - 1.1 * rng / 4
    # Shift by 1 to use only completed 1d bars (avoid look-ahead)
    h3_prev = np.roll(h3, 1)
    l3_prev = np.roll(l3, 1)
    h3_prev[0] = h3[0]
    l3_prev[0] = l3[0]
    
    # 1d EMA34 for trend filter
    ema34_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema34_slope = np.gradient(ema34_1d)  # slope of EMA34
    
    # Get 4h data for volume confirmation and ATR (primary timeframe)
    df_4h = get_htf_data(prices, '4h')
    volume_4h = df_4h['volume'].values
    high_4h = df_4h['high'].values
    low_4h = df_4h['low'].values
    close_4h = df_4h['close'].values
    
    # Volume average (20-period) on 4h
    volume_ma_20 = pd.Series(volume_4h).rolling(window=20, min_periods=20).mean().values
    
    # ATR (14-period) on 4h for stoploss
    tr1 = high_4h - low_4h
    tr2 = np.abs(high_4h - np.roll(close_4h, 1))
    tr3 = np.abs(low_4h - np.roll(close_4h, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]  # first period
    atr_14 = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    # Align all indicators to 4h timeframe (primary)
    h3_aligned = align_htf_to_ltf(prices, df_1d, h3_prev)
    l3_aligned = align_htf_to_ltf(prices, df_1d, l3_prev)
    ema34_slope_aligned = align_htf_to_ltf(prices, df_1d, ema34_slope)
    volume_ma_aligned = align_htf_to_ltf(prices, df_4h, volume_ma_20)
    atr_aligned = align_htf_to_ltf(prices, df_4h, atr_14)
    
    signals = np.zeros(n)
    position = 0  # -1: short, 0: flat, 1: long
    entry_price = 0.0
    atr_stop = 0.0
    
    start_idx = 100  # warmup
    
    for i in range(start_idx, n):
        if (np.isnan(h3_aligned[i]) or np.isnan(l3_aligned[i]) or 
            np.isnan(ema34_slope_aligned[i]) or np.isnan(volume_ma_aligned[i]) or 
            np.isnan(atr_aligned[i])):
            signals[i] = 0.0
            continue
        
        h3_val = h3_aligned[i]
        l3_val = l3_aligned[i]
        ema_slope = ema34_slope_aligned[i]
        vol_ma = volume_ma_aligned[i]
        atr_val = atr_aligned[i]
        vol = volume[i]
        price = close[i]
        
        if position == 0:
            # Look for breakouts with volume confirmation and trend filter
            # Long: price breaks above H3 + volume spike + EMA34 rising
            if price > h3_val and vol > 1.5 * vol_ma and ema_slope > 0:
                signals[i] = 0.25
                position = 1
                entry_price = price
                atr_stop = entry_price - 2.0 * atr_val
            # Short: price breaks below L3 + volume spike + EMA34 falling
            elif price < l3_val and vol > 1.5 * vol_ma and ema_slope < 0:
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

name = "4h_Camarilla_H3L3_1dEMA34_VolumeSpike_ATRTrail"
timeframe = "4h"
leverage = 1.0
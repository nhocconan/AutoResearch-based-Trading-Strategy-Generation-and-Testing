#!/usr/bin/env python3
"""
Hypothesis: 12h Camarilla pivot (R1/S1) breakout with 1d EMA34 trend filter and volume spike
- Uses 1d Camarilla levels (H3/L3) calculated from prior 1d candle
- Breakout above H3 (long) or below L3 (short) with volume > 1.5x 20-period MA
- 1d EMA34 slope confirms trend alignment (avoid counter-trend trades)
- ATR-based trailing stop (2.0x ATR) to manage risk and reduce drawdown
- Discrete position sizing: 0.0 (flat), ±0.25 (entry), ±0.125 (half-exit on strong adverse move)
- Designed for low trade frequency (~15-30/year) to minimize fee drag on 12h timeframe
- Works in bull markets (buying H3 breakouts in uptrend) and bear markets (selling L3 breakouts in downtrend)
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
    
    # Get 1d data for Camarilla pivots and EMA34 (HTF)
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    volume_1d = df_1d['volume'].values
    
    # Calculate prior 1d Camarilla levels (H3, L3) - using completed 1d bar only
    # Camarilla: H3 = close + 1.1*(high-low)/1.25, L3 = close - 1.1*(high-low)/1.25
    rng = high_1d - low_1d
    H3 = close_1d + (1.1 * rng / 1.25)
    L3 = close_1d - (1.1 * rng / 1.25)
    # Shift by 1 to use only completed 1d bars (avoid look-ahead)
    H3_shifted = np.roll(H3, 1)
    L3_shifted = np.roll(L3, 1)
    H3_shifted[0] = high_1d[0]  # fallback for first bar
    L3_shifted[0] = low_1d[0]
    
    # 1d EMA34 for trend filter
    ema34_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema34_slope = np.gradient(ema34_1d)  # slope of EMA34
    
    # Volume average (20-period) on 1d for confirmation
    volume_ma_20_1d = pd.Series(volume_1d).rolling(window=20, min_periods=20).mean().values
    
    # ATR (14-period) on 1d for stoploss
    tr1 = high_1d - low_1d
    tr2 = np.abs(high_1d - np.roll(close_1d, 1))
    tr3 = np.abs(low_1d - np.roll(close_1d, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]  # first period
    atr_14_1d = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    # Align all 1d indicators to 12h timeframe (primary)
    H3_aligned = align_htf_to_ltf(prices, df_1d, H3_shifted)
    L3_aligned = align_htf_to_ltf(prices, df_1d, L3_shifted)
    ema34_slope_aligned = align_htf_to_ltf(prices, df_1d, ema34_slope)
    volume_ma_aligned = align_htf_to_ltf(prices, df_1d, volume_ma_20_1d)
    atr_aligned = align_htf_to_ltf(prices, df_1d, atr_14_1d)
    
    signals = np.zeros(n)
    position = 0  # -1: short, 0: flat, 1: long
    entry_price = 0.0
    atr_stop = 0.0
    
    start_idx = 50  # warmup for 1d EMA34
    
    for i in range(start_idx, n):
        if (np.isnan(H3_aligned[i]) or np.isnan(L3_aligned[i]) or 
            np.isnan(ema34_slope_aligned[i]) or np.isnan(volume_ma_aligned[i]) or 
            np.isnan(atr_aligned[i])):
            signals[i] = 0.0
            continue
        
        H3 = H3_aligned[i]
        L3 = L3_aligned[i]
        ema_slope = ema34_slope_aligned[i]
        vol_ma = volume_ma_aligned[i]
        atr_val = atr_aligned[i]
        vol = volume[i]
        price = close[i]
        
        if position == 0:
            # Long: price breaks above H3 + volume spike + uptrend (EMA34 slope > 0)
            if price > H3 and vol > 1.5 * vol_ma and ema_slope > 0:
                signals[i] = 0.25
                position = 1
                entry_price = price
                atr_stop = entry_price - 2.0 * atr_val
            # Short: price breaks below L3 + volume spike + downtrend (EMA34 slope < 0)
            elif price < L3 and vol > 1.5 * vol_ma and ema_slope < 0:
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
                # Half-exit on strong adverse move (protect profits)
                if price < entry_price - 1.0 * atr_val:
                    signals[i] = 0.125  # reduce position
        
        elif position == -1:
            # Check stoploss
            if price >= atr_stop:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
                # Trail stop: lower stop if price moves favorably
                atr_stop = min(atr_stop, price + 1.5 * atr_val)
                # Half-exit on strong adverse move (protect profits)
                if price > entry_price + 1.0 * atr_val:
                    signals[i] = -0.125  # reduce position
    
    return signals

name = "12h_Camarilla_H3L3_1dEMA34_VolumeSpike_ATRTrail"
timeframe = "12h"
leverage = 1.0
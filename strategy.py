#!/usr/bin/env python3
"""
Hypothesis: 12h Camarilla R1/S1 breakout + 1d EMA34 trend filter + volume spike
- Uses 1d Camarilla pivot levels (R1, S1) calculated from prior 1d bar
- Entry on 12h close breaking above R1 (long) or below S1 (short) with volume > 1.5x 20-period MA
- Trend filter: 1d EMA34 slope must align with breakout direction (avoid counter-trend)
- ATR-based trailing stop (2.0x ATR) to manage risk and reduce drawdown
- Position size fixed at 0.25 to balance return and drawdown
- Designed for 12h timeframe to target 12-37 trades/year (50-150 total over 4 years)
- Works in bull markets (buying breakouts in uptrend) and bear markets (selling breakdowns in downtrend)
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
    
    # Calculate prior 1d Camarilla levels (using completed 1d bar)
    # Camarilla: R1 = close + 1.1*(high-low)/12, S1 = close - 1.1*(high-low)/12
    rng = high_1d - low_1d
    camarilla_r1 = close_1d + 1.1 * rng / 12
    camarilla_s1 = close_1d - 1.1 * rng / 12
    
    # 1d EMA34 for trend filter
    ema34_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema34_slope = np.gradient(ema34_1d)  # slope of EMA34
    
    # Align HTF indicators to 12h timeframe (primary)
    # Camarilla levels use prior completed 1d bar → already aligned by align_htf_to_ltf
    r1_aligned = align_htf_to_ltf(prices, df_1d, camarilla_r1)
    s1_aligned = align_htf_to_ltf(prices, df_1d, camarilla_s1)
    ema34_slope_aligned = align_htf_to_ltf(prices, df_1d, ema34_slope)
    
    # Volume confirmation: 20-period MA on 12h itself (no HTF needed)
    volume_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    # ATR (14-period) on 12h for stoploss
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]
    atr_14 = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    signals = np.zeros(n)
    position = 0  # -1: short, 0: flat, 1: long
    entry_price = 0.0
    atr_stop = 0.0
    
    start_idx = 50  # warmup for EMA34 and volume MA
    
    for i in range(start_idx, n):
        if (np.isnan(r1_aligned[i]) or np.isnan(s1_aligned[i]) or 
            np.isnan(ema34_slope_aligned[i]) or np.isnan(volume_ma_20[i]) or 
            np.isnan(atr_14[i])):
            signals[i] = 0.0
            continue
        
        r1 = r1_aligned[i]
        s1 = s1_aligned[i]
        ema_slope = ema34_slope_aligned[i]
        vol_ma = volume_ma_20[i]
        atr_val = atr_14[i]
        vol = volume[i]
        price = close[i]
        
        if position == 0:
            # Long: price breaks above R1 + volume spike + uptrend (EMA34 slope > 0)
            if price > r1 and vol > 1.5 * vol_ma and ema_slope > 0:
                signals[i] = 0.25
                position = 1
                entry_price = price
                atr_stop = entry_price - 2.0 * atr_val
            # Short: price breaks below S1 + volume spike + downtrend (EMA34 slope < 0)
            elif price < s1 and vol > 1.5 * vol_ma and ema_slope < 0:
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

name = "12h_Camarilla_R1S1_1dEMA34_VolumeSpike_ATRTrail"
timeframe = "12h"
leverage = 1.0
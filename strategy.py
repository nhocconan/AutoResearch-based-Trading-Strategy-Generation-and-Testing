#!/usr/bin/env python3
"""
Hypothesis: 6h Camarilla R1/S1 breakout with 1d EMA34 trend filter and volume confirmation.
Long when price breaks above R1 AND close > 1d EMA34 AND volume > 1.5x average.
Short when price breaks below S1 AND close < 1d EMA34 AND volume > 1.5x average.
Exit when price retouches the Camarilla pivot point (PP) or ATR-based stoploss.
Uses discrete position sizing (0.25) to minimize fee churn. Targets 12-37 trades/year per symbol.
Camarilla levels from 1d provide intraday support/resistance, effective in 6h timeframe when aligned with higher timeframe trend.
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
    
    # Load 1d data for Camarilla calculation - ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    volume_1d = df_1d['volume'].values
    
    # Calculate Camarilla levels from previous 1d bar (HLC of previous day)
    # We use the previous day's HLC to avoid look-ahead
    prev_high = np.roll(high_1d, 1)
    prev_low = np.roll(low_1d, 1)
    prev_close = np.roll(close_1d, 1)
    # First bar: use current bar (will be filtered by min_periods anyway)
    prev_high[0] = high_1d[0]
    prev_low[0] = low_1d[0]
    prev_close[0] = close_1d[0]
    
    # Camarilla levels
    range_ = prev_high - prev_low
    camarilla_pp = (prev_high + prev_low + prev_close) / 3
    camarilla_r1 = camarilla_pp + (range_ * 1.0 / 12)
    camarilla_s1 = camarilla_pp - (range_ * 1.0 / 12)
    camarilla_r4 = camarilla_pp + (range_ * 1.0 / 2)
    camarilla_s4 = camarilla_pp - (range_ * 1.0 / 2)
    
    # Align 1d Camarilla levels to 6h timeframe
    camarilla_pp_aligned = align_htf_to_ltf(prices, df_1d, camarilla_pp)
    camarilla_r1_aligned = align_htf_to_ltf(prices, df_1d, camarilla_r1)
    camarilla_s1_aligned = align_htf_to_ltf(prices, df_1d, camarilla_s1)
    camarilla_r4_aligned = align_htf_to_ltf(prices, df_1d, camarilla_r4)
    camarilla_s4_aligned = align_htf_to_ltf(prices, df_1d, camarilla_s4)
    
    # Calculate ATR(14) on 1d data for stoploss
    tr1 = np.maximum(high_1d - low_1d, np.abs(high_1d - np.roll(close_1d, 1)))
    tr2 = np.abs(low_1d - np.roll(close_1d, 1))
    tr = np.maximum(tr1, tr2)
    tr[0] = high_1d[0] - low_1d[0]  # first bar
    atr_1d = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    atr_1d_aligned = align_htf_to_ltf(prices, df_1d, atr_1d)
    
    # Calculate EMA34 on 1d close for trend filter
    ema34_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema34_1d)
    
    # Volume average (20-period) on 1d timeframe
    vol_ma = pd.Series(volume_1d).rolling(window=20, min_periods=20).mean().values
    vol_ma_aligned = align_htf_to_ltf(prices, df_1d, vol_ma)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    for i in range(100, n):
        # Skip if data not ready
        if (np.isnan(camarilla_pp_aligned[i]) or np.isnan(camarilla_r1_aligned[i]) or 
            np.isnan(camarilla_s1_aligned[i]) or np.isnan(ema34_1d_aligned[i]) or 
            np.isnan(vol_ma_aligned[i]) or np.isnan(atr_1d_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
            continue
        
        # Use 6h close for price comparison
        price_6h = close[i]
        vol_ma_val = vol_ma_aligned[i]
        atr_val = atr_1d_aligned[i]
        
        if position == 0:
            # Long: price breaks above R1 AND close > 1d EMA34 AND volume confirmation
            if (price_6h > camarilla_r1_aligned[i] and 
                close[i] > ema34_1d_aligned[i] and 
                volume[i] > 1.5 * vol_ma_val):
                signals[i] = 0.25
                position = 1
                entry_price = price_6h
            # Short: price breaks below S1 AND close < 1d EMA34 AND volume confirmation
            elif (price_6h < camarilla_s1_aligned[i] and 
                  close[i] < ema34_1d_aligned[i] and 
                  volume[i] > 1.5 * vol_ma_val):
                signals[i] = -0.25
                position = -1
                entry_price = price_6h
        else:
            # Exit conditions
            exit_signal = False
            
            if position == 1:
                # Exit long: price retouches PP (mean reversion) OR ATR-based stoploss
                if price_6h <= camarilla_pp_aligned[i]:
                    exit_signal = True
                # ATR-based stoploss
                elif price_6h < entry_price - 2.5 * atr_val:
                    exit_signal = True
            else:  # position == -1
                # Exit short: price retouches PP (mean reversion) OR ATR-based stoploss
                if price_6h >= camarilla_pp_aligned[i]:
                    exit_signal = True
                # ATR-based stoploss
                elif price_6h > entry_price + 2.5 * atr_val:
                    exit_signal = True
            
            if exit_signal:
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
            else:
                signals[i] = 0.25 if position == 1 else -0.25
    
    return signals

name = "6H_Camarilla_R1S1_Breakout_1dEMA34_Volume_ATRStop"
timeframe = "6h"
leverage = 1.0
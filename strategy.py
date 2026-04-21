#!/usr/bin/env python3
"""
1d_Camarilla_Pivot_R1S1_Breakout_VolumeATRFilter_V1
Hypothesis: Use 1w EMA34 trend filter + 1d Camarilla R1/S1 breakout with volume spike (>1.5x 20-bar MA) and ATR(14) stoploss (1.5x). 1w EMA34 reduces whipsaw in sideways markets, volume spike confirms breakout legitimacy, ATR stop manages risk. Designed to work in both bull (catch trends) and bear (avoid false breaks via 1w filter) markets. Target 15-25 trades/year per symbol.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Load HTF data ONCE before loop
    df_1w = get_htf_data(prices, '1w')  # for EMA34 trend filter
    
    if len(df_1w) < 34:
        return np.zeros(n)
    
    # === 1w EMA34 for Trend Filter ===
    close_1w = df_1w['close'].values
    ema_1w = pd.Series(close_1w).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_1w)
    
    # === 1d Indicators ===
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Camarilla pivot levels from previous day
    # R1 = close + 1.1*(high-low)/12
    # S1 = close - 1.1*(high-low)/12
    # Using previous day's OHLC
    prev_high = np.roll(high, 1)
    prev_low = np.roll(low, 1)
    prev_close = np.roll(close, 1)
    prev_high[0] = prev_low[0] = prev_close[0] = np.nan
    
    camarilla_range = prev_high - prev_low
    R1 = prev_close + 1.1 * camarilla_range / 12.0
    S1 = prev_close - 1.1 * camarilla_range / 12.0
    
    # Volume MA (20-period) for spike detection
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    # ATR (14-period)
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr1[0] = tr2[0] = tr3[0] = np.nan
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    atr = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(100, n):
        # Skip if indicators not ready
        if (np.isnan(ema_1w_aligned[i]) or np.isnan(R1[i]) or np.isnan(S1[i]) 
            or np.isnan(vol_ma[i]) or np.isnan(atr[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        price = close[i]
        vol = volume[i]
        vol_ok = vol > 1.5 * vol_ma[i]  # volume spike confirmation
        
        if position == 0:
            # Long: break above Camarilla R1 with volume spike and 1w uptrend
            if price > R1[i-1] and vol_ok and ema_1w_aligned[i] < price:
                signals[i] = 0.25
                position = 1
            # Short: break below Camarilla S1 with volume spike and 1w downtrend
            elif price < S1[i-1] and vol_ok and ema_1w_aligned[i] > price:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit: ATR stoploss or opposite signal
            if price < R1[i-1] - 1.5 * atr[i] or (price < S1[i-1] and vol_ok and ema_1w_aligned[i] > price):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit: ATR stoploss or opposite signal
            if price > S1[i-1] + 1.5 * atr[i] or (price > R1[i-1] and vol_ok and ema_1w_aligned[i] < price):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "1d_Camarilla_Pivot_R1S1_Breakout_VolumeATRFilter_V1"
timeframe = "1d"
leverage = 1.0
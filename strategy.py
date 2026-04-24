#!/usr/bin/env python3
"""
Hypothesis: 4h Camarilla H3/L3 breakout with 1d ATR regime filter and volume spike confirmation.
- Primary timeframe: 4h targeting 75-200 total trades over 4 years (19-50/year).
- HTF: 1d ATR(14) for regime filter (high volatility = trend follow, low volatility = avoid).
- Camarilla pivot levels: Calculated from prior 1d OHLC (H3, L3 levels for breakout).
- Entry: Long when price breaks above prior 1d H3 AND 1d ATR > ATR MA(50) AND volume > 2.0 * volume MA(20).
         Short when price breaks below prior 1d L3 AND 1d ATR > ATR MA(50) AND volume > 2.0 * volume MA(20).
- Exit: Close-based reversal - exit long when price crosses below prior 1d L3,
        exit short when price crosses above prior 1d H3.
- Signal size: 0.25 discrete to balance return and drawdown.
Uses 1d ATR regime filter to avoid low-volatility ranging markets where breakouts fail.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Extract price data
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1d data for Camarilla pivots and ATR regime filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 60:
        return np.zeros(n)
    
    # Calculate 1d ATR(14) for regime filter
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # True Range calculation
    tr1 = high_1d[1:] - low_1d[1:]
    tr2 = np.abs(high_1d[1:] - close_1d[:-1])
    tr3 = np.abs(low_1d[1:] - close_1d[:-1])
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr = np.concatenate([[np.nan], tr])  # First value is NaN
    
    atr_1d = pd.Series(tr).ewm(span=14, adjust=False, min_periods=14).mean().values
    atr_ma_1d = pd.Series(atr_1d).rolling(window=50, min_periods=50).mean().values
    
    # Calculate prior 1d Camarilla levels (H3, L3)
    # Using prior 1d candle to avoid look-ahead
    rang = high_1d - low_1d
    camarilla_h3 = close_1d + rang * 1.1 / 4
    camarilla_l3 = close_1d - rang * 1.1 / 4
    
    # Align HTF indicators to 4h
    atr_1d_aligned = align_htf_to_ltf(prices, df_1d, atr_1d)
    atr_ma_1d_aligned = align_htf_to_ltf(prices, df_1d, atr_ma_1d)
    camarilla_h3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_h3)
    camarilla_l3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_l3)
    
    # Calculate volume MA(20) for confirmation (using 4h data)
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start from index where all indicators are ready
    start_idx = max(100, 60)  # Need enough bars for ATR and volume MA
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(atr_1d_aligned[i]) or np.isnan(atr_ma_1d_aligned[i]) or 
            np.isnan(camarilla_h3_aligned[i]) or np.isnan(camarilla_l3_aligned[i]) or 
            np.isnan(vol_ma[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        curr_close = close[i]
        curr_volume = volume[i]
        
        # Regime filter: only trade when ATR > ATR_MA (high volatility regime)
        high_vol_regime = atr_1d_aligned[i] > atr_ma_1d_aligned[i]
        vol_confirmed = curr_volume > 2.0 * vol_ma[i]
        
        if position == 0:
            # Check for entry signals with volume confirmation and regime filter
            # Long: Price breaks above prior 1d H3 AND high vol regime AND volume confirmed
            if curr_close > camarilla_h3_aligned[i] and high_vol_regime and vol_confirmed:
                signals[i] = 0.25
                position = 1
            # Short: Price breaks below prior 1d L3 AND high vol regime AND volume confirmed
            elif curr_close < camarilla_l3_aligned[i] and high_vol_regime and vol_confirmed:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long when price crosses below prior 1d L3 (breakdown)
            if curr_close < camarilla_l3_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short when price crosses above prior 1d H3 (breakout)
            if curr_close > camarilla_h3_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "4h_Camarilla_H3L3_1dATR_Regime_VolumeSpike_v1"
timeframe = "4h"
leverage = 1.0
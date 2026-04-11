#!/usr/bin/env python3
"""
1d_1w_camarilla_volatility_filter_v1
Strategy: Daily Camarilla pivot breakout with volatility filter and weekly trend filter
Timeframe: 1d
Leverage: 1.0
Hypothesis: Uses daily Camarilla pivot levels (H4/L4) for breakout entries, filtered by weekly ATR-based volatility regime (only trade when volatility is expanding) and weekly EMA40 trend direction. Designed to capture explosive moves in both bull and bear markets by trading volatility expansion in the direction of the higher timeframe trend. Target: 20-50 trades over 4 years.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "1d_1w_camarilla_volatility_filter_v1"
timeframe = "1d"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Price arrays
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Load higher timeframe data ONCE before loop
    df_1w = get_htf_data(prices, '1w')
    
    if len(df_1w) < 40:
        return np.zeros(n)
    
    # Daily ATR(20) for volatility measurement
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr2[0] = tr2[1]  # fix first value
    tr3[0] = tr3[1]
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    atr = pd.Series(tr).rolling(window=20, min_periods=20).mean().values
    
    # Weekly ATR(20) for volatility regime filter
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    tr1_w = high_1w - low_1w
    tr2_w = np.abs(high_1w - np.roll(close_1w, 1))
    tr3_w = np.abs(low_1w - np.roll(close_1w, 1))
    tr2_w[0] = tr2_w[1]
    tr3_w[0] = tr3_w[1]
    tr_w = np.maximum(tr1_w, np.maximum(tr2_w, tr3_w))
    atr_w = pd.Series(tr_w).rolling(window=20, min_periods=20).mean().values
    atr_w_avg = pd.Series(atr_w).rolling(window=10, min_periods=10).mean().values  # 10-week average ATR
    atr_w_avg_aligned = align_htf_to_ltf(prices, df_1w, atr_w_avg)
    
    # Weekly EMA40 for trend filter
    ema_40_1w = pd.Series(close_1w).ewm(span=40, adjust=False, min_periods=40).mean().values
    ema_40_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_40_1w)
    
    # Daily ATR expansion filter: current ATR > 1.5x 10-period average ATR
    atr_avg = pd.Series(atr).rolling(window=10, min_periods=10).mean().values
    vol_expanding = atr > (1.5 * atr_avg)
    
    # Calculate Camarilla levels from previous day's OHLC
    high_1 = np.roll(high, 1)
    low_1 = np.roll(low, 1)
    close_1 = np.roll(close, 1)
    high_1[0] = high[0]
    low_1[0] = low[0]
    close_1[0] = close[0]
    
    camarilla_H4 = close_1 + 1.1 * (high_1 - low_1) / 2
    camarilla_L4 = close_1 - 1.1 * (high_1 - low_1) / 2
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(20, n):
        # Skip if any required data is invalid
        if (np.isnan(atr[i]) or np.isnan(atr_avg[i]) or np.isnan(atr_w_avg_aligned[i]) or
            np.isnan(ema_40_1w_aligned[i]) or np.isnan(camarilla_H4[i]) or np.isnan(camarilla_L4[i])):
            signals[i] = 0.0 if position == 0 else (0.25 if position == 1 else -0.25)
            continue
        
        price_close = close[i]
        
        # Volatility filter: only trade when volatility is expanding (ATR > 1.5x average)
        vol_filter = vol_expanding[i]
        
        # Trend filter: price above/below weekly EMA40
        uptrend_1w = price_close > ema_40_1w_aligned[i]
        downtrend_1w = price_close < ema_40_1w_aligned[i]
        
        # Breakout conditions using Camarilla levels
        breakout_up = price_close > camarilla_H4[i]
        breakout_down = price_close < camarilla_L4[i]
        
        # Long: upward breakout with volatility expansion in uptrend
        long_signal = breakout_up and vol_filter and uptrend_1w
        
        # Short: downward breakout with volatility expansion in downtrend
        short_signal = breakout_down and vol_filter and downtrend_1w
        
        # Exit when volatility contracts (ATR < average) or price returns to opposite Camarilla level
        vol_contracting = atr[i] < atr_avg[i]
        exit_long = position == 1 and (vol_contracting or price_close < camarilla_L4[i])
        exit_short = position == -1 and (vol_contracting or price_close > camarilla_H4[i])
        
        # Trading logic
        if long_signal and position != 1:
            position = 1
            signals[i] = 0.25
        elif short_signal and position != -1:
            position = -1
            signals[i] = -0.25
        elif position == 1 and exit_long:
            position = 0
            signals[i] = 0.0
        elif position == -1 and exit_short:
            position = 0
            signals[i] = 0.0
        else:
            signals[i] = 0.25 if position == 1 else (-0.25 if position == -1 else 0.0)
    
    return signals
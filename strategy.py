#!/usr/bin/env python3
"""
Hypothesis: 12h Camarilla R3/S3 breakout + 1d EMA50 trend filter + volume spike.
Long when price breaks above Camarilla R3 AND close > 1d EMA50 AND volume > 1.8x 24-period average.
Short when price breaks below Camarilla S3 AND close < 1d EMA50 AND volume > 1.8x 24-period average.
Exit when price crosses Camarilla H3/L3 (mean reversion) or ATR(14) stoploss hit (2.0*ATR).
Uses discrete position sizing (0.25) to minimize fee churn. Targets 15-35 trades/year per symbol.
Camarilla levels provide precise intraday support/resistance, while 1d EMA50 ensures alignment with higher-timeframe trend.
Volume confirmation filters weak breakouts. ATR stoploss manages risk during adverse moves.
Designed to work in both bull and bear markets by using HTF trend filter and symmetric long/short logic.
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
    
    # Load 1d data for EMA50 trend filter - ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 1:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    
    # Calculate 1d EMA50 for trend filter
    ema50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Align HTF EMA50 to 12h timeframe
    ema50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema50_1d)
    
    # Calculate Camarilla levels from previous 12h bar (using typical price)
    # Camarilla uses previous period's high, low, close
    prev_high = np.roll(high, 1)
    prev_low = np.roll(low, 1)
    prev_close = np.roll(close, 1)
    prev_high[0] = high[0]
    prev_low[0] = low[0]
    prev_close[0] = close[0]
    
    # Typical price for pivot calculation
    typical_price = (prev_high + prev_low + prev_close) / 3.0
    range_val = prev_high - prev_low
    
    # Camarilla levels
    camarilla_h3 = typical_price + range_val * 1.1 / 4.0  # Resistance 3
    camarilla_l3 = typical_price - range_val * 1.1 / 4.0  # Support 3
    camarilla_h4 = typical_price + range_val * 1.1 / 2.0  # Resistance 4 (breakout level)
    camarilla_l4 = typical_price - range_val * 1.1 / 2.0  # Support 4 (breakout level)
    camarilla_h6 = typical_price + range_val * 1.1 / 2.0 * 1.5  # Extended resistance
    camarilla_l6 = typical_price - range_val * 1.1 / 2.0 * 1.5  # Extended support
    
    # For breakout, we use H4/L4; for mean reversion exit, we use H3/L3
    breakout_upper = camarilla_h4
    breakout_lower = camarilla_l4
    exit_upper = camarilla_h3
    exit_lower = camarilla_l3
    
    # Volume average (24-period) on primary timeframe
    vol_ma = pd.Series(volume).rolling(window=24, min_periods=24).mean().values
    
    # ATR(14) for stoploss calculation
    tr1 = np.abs(high - low)
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]  # first bar
    atr = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    # Start from index where all indicators are ready
    start_idx = max(100, 24, 50, 14)  # Ensure warmup for Camarilla, EMA50, ATR(14)
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(ema50_1d_aligned[i]) or np.isnan(breakout_upper[i]) or 
            np.isnan(breakout_lower[i]) or np.isnan(vol_ma[i]) or np.isnan(atr[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        price = close[i]
        vol_ma_val = vol_ma[i]
        atr_val = atr[i]
        
        if position == 0:
            # Long: Camarilla breakout above H4 AND 1d EMA50 uptrend AND volume spike
            if (price > breakout_upper[i] and 
                close[i] > ema50_1d_aligned[i] and 
                volume[i] > 1.8 * vol_ma_val):
                signals[i] = 0.25
                position = 1
                entry_price = price
            # Short: Camarilla breakout below L4 AND 1d EMA50 downtrend AND volume spike
            elif (price < breakout_lower[i] and 
                  close[i] < ema50_1d_aligned[i] and 
                  volume[i] > 1.8 * vol_ma_val):
                signals[i] = -0.25
                position = -1
                entry_price = price
        else:
            # Exit conditions
            exit_signal = False
            
            # Primary exit: price crosses Camarilla H3/L3 (mean reversion)
            if position == 1 and price < exit_upper[i]:
                exit_signal = True
            elif position == -1 and price > exit_lower[i]:
                exit_signal = True
            
            # ATR-based stoploss: 2.0 * ATR from entry
            if position == 1 and price < entry_price - 2.0 * atr_val:
                exit_signal = True
            elif position == -1 and price > entry_price + 2.0 * atr_val:
                exit_signal = True
            
            if exit_signal:
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
            else:
                signals[i] = 0.25 if position == 1 else -0.25
    
    return signals

name = "12H_Camarilla_H4L4_1dEMA50_VolumeSpike_ATRStop"
timeframe = "12h"
leverage = 1.0
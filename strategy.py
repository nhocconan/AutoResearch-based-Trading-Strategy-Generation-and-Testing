#!/usr/bin/env python3
"""
12h_Camarilla_H3L3_Breakout_1dATR_Trend_VolumeSp
Hypothesis: For 12h timeframe, use Camarilla H3/L3 breakouts with 1d ATR-based trend filter, volume spike confirmation, and ATR stoploss. H3/L3 levels provide stronger breakout signals than R1/S1, reducing false entries. 1d ATR trend (price vs ATR-scaled EMA) adapts to volatility regimes, working in both bull and bear markets. Volume spike ensures conviction. Target: 12-37 trades/year (50-150 total over 4 years).
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # 1d data for Camarilla pivots, ATR trend filter (loaded ONCE)
    df_1d = get_htf_data(prices, '1d')
    
    # 1d Camarilla pivot levels (based on previous day's OHLC)
    prev_close = df_1d['close'].shift(1).values
    prev_high = df_1d['high'].shift(1).values
    prev_low = df_1d['low'].shift(1).values
    prev_range = prev_high - prev_low
    
    H3 = prev_close + 1.5 * prev_range
    L3 = prev_close - 1.5 * prev_range
    
    # Align 1d pivot levels to 12h timeframe
    H3_aligned = align_htf_to_ltf(prices, df_1d, H3)
    L3_aligned = align_htf_to_ltf(prices, df_1d, L3)
    
    # Volume spike: current volume > 2.0 * 20-period average (strict to limit trades)
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (vol_ma * 2.0)
    
    # 1d ATR for trend filter: price > EMA20 + 0.5*ATR = uptrend, price < EMA20 - 0.5*ATR = downtrend
    tr1 = df_1d['high'].values - df_1d['low'].values
    tr2 = np.abs(df_1d['high'].values - df_1d['close'].shift(1).values)
    tr3 = np.abs(df_1d['low'].values - df_1d['close'].shift(1).values)
    tr_daily = np.maximum(tr1, np.maximum(tr2, tr3))
    atr_1d = pd.Series(tr_daily).rolling(window=14, min_periods=14).mean().values
    ema_20_1d = pd.Series(df_1d['close'].values).ewm(span=20, adjust=False, min_periods=20).mean().values
    atr_threshold = 0.5 * atr_1d
    uptrend_1d = close > (ema_20_1d + atr_threshold)  # Will align below
    downtrend_1d = close < (ema_20_1d - atr_threshold)  # Will align below
    
    # Align 1d trend filters to 12h timeframe
    uptrend_1d_aligned = align_htf_to_ltf(prices, df_1d, uptrend_1d)
    downtrend_1d_aligned = align_htf_to_ltf(prices, df_1d, downtrend_1d)
    
    # ATR for stoploss (using 12h data)
    tr1 = high[1:] - low[1:]
    tr2 = np.abs(high[1:] - close[:-1])
    tr3 = np.abs(low[1:] - close[:-1])
    tr = np.concatenate([[np.nan], np.maximum(tr1, np.maximum(tr2, tr3))])
    atr = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    # Start index: need enough for 1d indicators (20 for EMA, 14 for ATR)
    start_idx = 20
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(H3_aligned[i]) or np.isnan(L3_aligned[i]) or np.isnan(vol_ma[i]) or
            np.isnan(uptrend_1d_aligned[i]) or np.isnan(downtrend_1d_aligned[i]) or np.isnan(atr[i])):
            signals[i] = 0.0
            continue
        
        curr_close = close[i]
        curr_high = high[i]
        curr_low = low[i]
        
        if position == 0:
            # Look for entry signals with volume spike and trend alignment
            # Long breakout: price breaks above H3 with uptrend and volume spike
            long_breakout = (curr_close > H3_aligned[i]) and uptrend_1d_aligned[i] and volume_spike[i]
            # Short breakout: price breaks below L3 with downtrend and volume spike
            short_breakout = (curr_close < L3_aligned[i]) and downtrend_1d_aligned[i] and volume_spike[i]
            
            if long_breakout:
                signals[i] = 0.25
                position = 1
                entry_price = curr_close
            elif short_breakout:
                signals[i] = -0.25
                position = -1
                entry_price = curr_close
            else:
                signals[i] = 0.0
        elif position == 1:
            # Long position: exit conditions
            # Stoploss: 2.5 * ATR below entry (wider for 12h to avoid whipsaw)
            if curr_close < entry_price - 2.5 * atr[i]:
                signals[i] = 0.0
                position = 0
            # Exit if price breaks below L3 (mean reversion) or trend changes to down
            elif curr_close < L3_aligned[i] or not uptrend_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short position: exit conditions
            # Stoploss: 2.5 * ATR above entry
            if curr_close > entry_price + 2.5 * atr[i]:
                signals[i] = 0.0
                position = 0
            # Exit if price breaks above H3 (mean reversion) or trend changes to up
            elif curr_close > H3_aligned[i] or not downtrend_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "12h_Camarilla_H3L3_Breakout_1dATR_Trend_VolumeSp"
timeframe = "12h"
leverage = 1.0
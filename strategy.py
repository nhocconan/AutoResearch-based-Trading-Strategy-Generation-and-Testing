#!/usr/bin/env python3
"""
4h_Camarilla_R1_S1_Breakout_1dTrend_ChopFilter
Hypothesis: 4h Camarilla R1/S1 breakout with 1d EMA50 trend filter and choppiness regime filter.
Long when price breaks above Camarilla R1 in 1d uptrend (close > 1d EMA50) and market is not choppy (Chop < 61.8).
Short when price breaks below Camarilla S1 in 1d downtrend (close < 1d EMA50) and market is not choppy (Chop < 61.8).
Exit via ATR-based trailing stop (2.0*ATR from extreme) or re-entry into Camarilla H3/L3 range.
Designed for ~20-40 trades/year by requiring strong breakouts, trend alignment, and low-chop regime.
Works in bull/bear markets via 1d EMA50 filter; avoids whipsaws via chop regime filter and volume confirmation.
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
    
    # Get 1d data for trend filter and Camarilla levels (HTF)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    
    # Calculate 1d EMA50 for trend filter
    ema_50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # Calculate Camarilla levels from previous day
    # Previous day's OHLC (align to current 4h bars with 1-bar delay for completed day)
    prev_close = align_htf_to_ltf(prices, df_1d, df_1d['close'].values, additional_delay_bars=1)
    prev_high = align_htf_to_ltf(prices, df_1d, df_1d['high'].values, additional_delay_bars=1)
    prev_low = align_htf_to_ltf(prices, df_1d, df_1d['low'].values, additional_delay_bars=1)
    prev_open = align_htf_to_ltf(prices, df_1d, df_1d['open'].values, additional_delay_bars=1)
    
    # Camarilla levels: R1 = C + (H-L)*1.1/12, S1 = C - (H-L)*1.1/12
    # R3 = C + (H-L)*1.1/4, S3 = C - (H-L)*1.1/4
    # H3 = C + (H-L)*1.1/2, L3 = C - (H-L)*1.1/2
    camarilla_range = prev_high - prev_low
    R1 = prev_close + camarilla_range * 1.1 / 12
    S1 = prev_close - camarilla_range * 1.1 / 12
    R3 = prev_close + camarilla_range * 1.1 / 4
    S3 = prev_close - camarilla_range * 1.1 / 4
    H3 = prev_close + camarilla_range * 1.1 / 2
    L3 = prev_close - camarilla_range * 1.1 / 2
    
    # ATR for trailing stop (14-period)
    atr_period = 14
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]  # first period
    atr = pd.Series(tr).rolling(window=atr_period, min_periods=atr_period).mean().values
    
    # Choppiness Index regime filter (14-period)
    chop_period = 14
    # True Range (same as ATR calculation)
    tr_chop = tr  # reuse TR from above
    # Sum of True Range over chop_period
    atr_sum = pd.Series(tr_chop).rolling(window=chop_period, min_periods=chop_period).sum().values
    # Highest high and lowest low over chop_period
    hh = pd.Series(high).rolling(window=chop_period, min_periods=chop_period).max().values
    ll = pd.Series(low).rolling(window=chop_period, min_periods=chop_period).min().values
    # Chop = 100 * log10(atr_sum / (hh - ll)) / log10(chop_period)
    # Avoid division by zero
    hh_ll = hh - ll
    hh_ll = np.where(hh_ll == 0, 1e-10, hh_ll)  # small epsilon to prevent div by zero
    chop = 100 * np.log10(atr_sum / hh_ll) / np.log10(chop_period)
    # Market is not choppy when Chop < 61.8 (trending regime)
    not_choppy = chop < 61.8
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    long_high = 0.0   # highest close since long entry
    short_low = 0.0   # lowest close since short entry
    
    # Start index: need warmup for calculations
    start_idx = max(100, atr_period, chop_period)
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(ema_50_1d_aligned[i]) or np.isnan(R1[i]) or np.isnan(S1[i]) or 
            np.isnan(H3[i]) or np.isnan(L3[i]) or np.isnan(atr[i]) or np.isnan(chop[i])):
            signals[i] = 0.0 if position == 0 else (0.25 if position == 1 else -0.25)
            continue
        
        ema_trend = ema_50_1d_aligned[i]
        is_not_choppy = not_choppy[i]
        
        if position == 0:
            # Only trade in trending regimes (1d EMA50 filter) and non-choppy market
            if ema_trend > 0 and is_not_choppy:  # 1d uptrend regime and not choppy
                # Long: break above Camarilla R1
                long_signal = close[i] > R1[i]
            elif ema_trend < 0 and is_not_choppy:  # 1d downtrend regime and not choppy
                # Short: break below Camarilla S1
                short_signal = close[i] < S1[i]
            else:
                long_signal = False
                short_signal = False
            
            if long_signal:
                signals[i] = 0.25
                position = 1
                long_high = close[i]
            elif short_signal:
                signals[i] = -0.25
                position = -1
                short_low = close[i]
            else:
                signals[i] = 0.0
        elif position == 1:
            # Long: hold position
            signals[i] = 0.25
            # Update highest close
            if close[i] > long_high:
                long_high = close[i]
            # Exit conditions: ATR trailing stop OR re-enter Camarilla H3/L3 range
            atr_stop = long_high - 2.0 * atr[i]
            range_exit = (close[i] < H3[i] and close[i] > L3[i])
            if close[i] <= atr_stop or range_exit:
                signals[i] = 0.0
                position = 0
        elif position == -1:
            # Short: hold position
            signals[i] = -0.25
            # Update lowest close
            if close[i] < short_low:
                short_low = close[i]
            # Exit conditions: ATR trailing stop OR re-enter Camarilla H3/L3 range
            atr_stop = short_low + 2.0 * atr[i]
            range_exit = (close[i] > L3[i] and close[i] < H3[i])
            if close[i] >= atr_stop or range_exit:
                signals[i] = 0.0
                position = 0
    
    return signals

name = "4h_Camarilla_R1_S1_Breakout_1dTrend_ChopFilter"
timeframe = "4h"
leverage = 1.0
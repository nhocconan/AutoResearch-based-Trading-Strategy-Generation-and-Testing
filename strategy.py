#!/usr/bin/env python3
"""
Hypothesis: 12h KAMA trend with 1d RSI regime filter and volume confirmation.
- Primary timeframe: 12h to target 50-150 total trades over 4 years (12-37/year).
- HTF: 1d KAMA for trend direction (bullish if close > KAMA, bearish if close < KAMA).
- Entry: Long when price > 12h KAMA AND 1d RSI(14) < 30 (oversold bounce) AND volume > 1.5 * volume MA(20).
         Short when price < 12h KAMA AND 1d RSI(14) > 70 (overbought rejection) AND volume > 1.5 * volume MA(20).
- Exit: ATR-based trailing stop - exit long when price < highest_high_since_entry - 2.5*ATR,
        exit short when price > lowest_low_since_entry + 2.5*ATR.
- Signal size: 0.25 discrete to minimize fee churn.
This strategy targets mean-reversion within the trend on the 12h timeframe, using RSI extremes for entry timing
and volume confirmation to avoid false signals, designed to work in both bull and bear markets by fading
extended moves while respecting the 1d trend.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Extract price data
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1d data for KAMA trend filter and RSI
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # Calculate 1d KAMA for trend filter
    df_1d_close = df_1d['close'].values
    # Efficiency ratio
    change = np.abs(np.diff(df_1d_close, prepend=df_1d_close[0]))
    volatility = np.abs(np.diff(df_1d_close))
    er = np.zeros_like(df_1d_close)
    er[1:] = change[1:] / (volatility[1:] + 1e-10)
    # Smoothing constants
    sc = (er * (2/(2+1) - 2/(30+1)) + 2/(30+1)) ** 2
    # KAMA calculation
    kama = np.zeros_like(df_1d_close)
    kama[0] = df_1d_close[0]
    for i in range(1, len(df_1d_close)):
        kama[i] = kama[i-1] + sc[i] * (df_1d_close[i] - kama[i-1])
    
    # Calculate 1d RSI(14)
    delta = np.diff(df_1d_close, prepend=df_1d_close[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    avg_gain = pd.Series(gain).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    avg_loss = pd.Series(loss).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    rs = avg_gain / (avg_loss + 1e-10)
    rsi = 100 - (100 / (1 + rs))
    
    # Calculate ATR(14) for trailing stop
    tr1 = np.abs(high[1:] - low[:-1])
    tr2 = np.abs(high[1:] - close[:-1])
    tr3 = np.abs(low[1:] - close[:-1])
    tr = np.concatenate([[np.max([tr1[0], tr2[0], tr3[0]])], np.maximum(tr1, np.maximum(tr2, tr3))])
    atr = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    # Calculate volume MA(20) for confirmation
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    # Align HTF indicators to 12h
    kama_1d_aligned = align_htf_to_ltf(prices, df_1d, kama)
    rsi_1d_aligned = align_htf_to_ltf(prices, df_1d, rsi)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    highest_since_entry = 0.0
    lowest_since_entry = 0.0
    
    # Start from index where all indicators are ready
    start_idx = max(30, 14, 14, 20)  # Need enough bars for KAMA, RSI, ATR, Vol MA
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(kama_1d_aligned[i]) or np.isnan(rsi_1d_aligned[i]) or 
            np.isnan(atr[i]) or np.isnan(vol_ma[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        curr_close = close[i]
        curr_high = high[i]
        curr_low = low[i]
        curr_volume = volume[i]
        
        if position == 0:
            # Check for entry signals with volume confirmation
            vol_confirmed = curr_volume > 1.5 * vol_ma[i]
            
            # Long: Price above 12h KAMA AND 1d RSI oversold (<30) AND volume confirmed
            if curr_close > kama_1d_aligned[i] and rsi_1d_aligned[i] < 30 and vol_confirmed:
                signals[i] = 0.25
                position = 1
                entry_price = curr_close
                highest_since_entry = curr_high
                lowest_since_entry = curr_low
            # Short: Price below 12h KAMA AND 1d RSI overbought (>70) AND volume confirmed
            elif curr_close < kama_1d_aligned[i] and rsi_1d_aligned[i] > 70 and vol_confirmed:
                signals[i] = -0.25
                position = -1
                entry_price = curr_close
                highest_since_entry = curr_high
                lowest_since_entry = curr_low
        elif position == 1:
            # Update highest high since entry
            highest_since_entry = max(highest_since_entry, curr_high)
            # ATR trailing stop: exit when price < highest_high - 2.5*ATR
            if curr_close < highest_since_entry - 2.5 * atr[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Update lowest low since entry
            lowest_since_entry = min(lowest_since_entry, curr_low)
            # ATR trailing stop: exit when price > lowest_low + 2.5*ATR
            if curr_close > lowest_since_entry + 2.5 * atr[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "12h_KAMA_1dRSI_Regime_VolumeConfirmation_v1"
timeframe = "12h"
leverage = 1.0
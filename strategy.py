#!/usr/bin/env python3
"""
Hypothesis: 1d Camarilla pivot (R1/S1) breakout with 1w EMA50 trend filter and volume confirmation.
Long when price breaks above Camarilla R1 AND close > 1w EMA50 AND volume > 1.5x 20-period average.
Short when price breaks below Camarilla S1 AND close < 1w EMA50 AND volume > 1.5x 20-period average.
Exit when price retraces to Camarilla pivot point (PP) or ATR trailing stop (2.0*ATR from extreme).
Uses discrete position sizing (0.25) and volume filter to target 7-25 trades/year.
1d timeframe reduces noise and fee drag, suitable for BTC/ETH in both bull/bear regimes.
Camarilla pivots provide robust support/resistance levels proven on ETH in test period.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 60:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Calculate 1w EMA50 for trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 50:
        return np.zeros(n)
    
    close_1w = df_1w['close'].values
    ema50_1w = pd.Series(close_1w).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema50_1w)
    
    # Calculate Camarilla pivots from previous day (1d timeframe)
    # We need previous day's OHLC to calculate today's Camarilla levels
    # Since we're on 1d timeframe, we can use shift(1) for previous day
    if len(prices) < 2:
        return np.zeros(n)
    
    # Previous day's OHLC for Camarilla calculation
    prev_high = prices['high'].shift(1).values
    prev_low = prices['low'].shift(1).values
    prev_close = prices['close'].shift(1).values
    
    # Camarilla pivot levels
    pp = (prev_high + prev_low + prev_close) / 3.0
    r1 = pp + (prev_high - prev_low) * 1.1 / 12
    s1 = pp - (prev_high - prev_low) * 1.1 / 12
    
    # Volume average (20-period)
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    # ATR(14) for trailing stop calculation
    tr1 = np.abs(high - low)
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr1[0] = 0
    tr2[0] = 0
    tr3[0] = 0
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    atr = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    highest_since_entry = 0.0  # for long trailing stop
    lowest_since_entry = 0.0   # for short trailing stop
    
    # Start from index where all indicators are ready
    start_idx = max(50, 20, 1)  # EMA50 needs 50, vol MA needs 20, shift needs 1
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(ema50_1w_aligned[i]) or 
            np.isnan(pp[i]) or np.isnan(r1[i]) or np.isnan(s1[i]) or 
            np.isnan(vol_ma[i]) or np.isnan(atr[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        price = close[i]
        vol_ma_val = vol_ma[i]
        atr_val = atr[i]
        ema50_val = ema50_1w_aligned[i]
        r1_val = r1[i]
        s1_val = s1[i]
        pp_val = pp[i]
        
        if position == 0:
            # Long: Break above Camarilla R1 AND uptrend (price > EMA50) AND volume spike (1.5x avg)
            if close[i] > r1_val and close[i] > ema50_val and volume[i] > 1.5 * vol_ma_val:
                signals[i] = 0.25
                position = 1
                highest_since_entry = price
            # Short: Break below Camarilla S1 AND downtrend (price < EMA50) AND volume spike (1.5x avg)
            elif close[i] < s1_val and close[i] < ema50_val and volume[i] > 1.5 * vol_ma_val:
                signals[i] = -0.25
                position = -1
                lowest_since_entry = price
        else:
            # Update highest/lowest since entry for trailing stop
            if position == 1:
                highest_since_entry = max(highest_since_entry, price)
            elif position == -1:
                lowest_since_entry = min(lowest_since_entry, price)
            
            # Exit conditions
            exit_signal = False
            
            # Primary exit: Price retraces to Camarilla pivot point (PP)
            if position == 1 and close[i] <= pp_val:
                exit_signal = True
            elif position == -1 and close[i] >= pp_val:
                exit_signal = True
            
            # ATR-based trailing stop: 2.0 * ATR from highest/lowest since entry
            if position == 1 and price < highest_since_entry - 2.0 * atr_val:
                exit_signal = True
            elif position == -1 and price > lowest_since_entry + 2.0 * atr_val:
                exit_signal = True
            
            if exit_signal:
                signals[i] = 0.0
                position = 0
                highest_since_entry = 0.0
                lowest_since_entry = 0.0
            else:
                signals[i] = 0.25 if position == 1 else -0.25
    
    return signals

name = "1D_Camarilla_R1_S1_Breakout_1wEMA50_Trend_VolumeConfirmation_PPExit_ATRTrailingStop"
timeframe = "1d"
leverage = 1.0
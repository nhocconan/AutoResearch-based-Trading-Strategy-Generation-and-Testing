#!/usr/bin/env python3
"""
1d_Camarilla_R1S1_Breakout_Volume_ATRFilter_v1
Hypothesis: Breakout of Camarilla R1/S1 levels on daily timeframe with volume confirmation and ATR-based stoploss.
Works in bull/bear markets: Breakouts capture momentum in trending markets, while volume filter avoids false signals.
ATR stoploss manages risk during reversals. Uses 1w EMA for long-term trend bias to improve win rate.
Target: 15-25 trades/year per symbol (60-100 over 4 years).
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Load 1d data once for Camarilla levels and ATR
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Previous day's OHLC for Camarilla calculation
    prev_high = np.roll(high_1d, 1)
    prev_low = np.roll(low_1d, 1)
    prev_close = np.roll(close_1d, 1)
    prev_high[0] = np.nan
    prev_low[0] = np.nan
    prev_close[0] = np.nan
    
    # Camarilla levels: R1, S1 (breakout levels)
    rang = prev_high - prev_low
    r1 = prev_close + rang * 1.0 / 12
    s1 = prev_close - rang * 1.0 / 12
    
    # Align to 1d timeframe (same as primary, so no shift needed but using align for consistency)
    r1_aligned = align_htf_to_ltf(prices, df_1d, r1)
    s1_aligned = align_htf_to_ltf(prices, df_1d, s1)
    
    # ATR(14) for stoploss and volatility filter
    tr1 = np.maximum(high_1d[1:] - low_1d[1:], np.abs(high_1d[1:] - close_1d[:-1]))
    tr1 = np.maximum(tr1, np.abs(low_1d[1:] - close_1d[:-1]))
    tr1 = np.concatenate([[np.nan], tr1])
    atr_1d = pd.Series(tr1).ewm(span=14, adjust=False, min_periods=14).mean().values
    atr_1d_aligned = align_htf_to_ltf(prices, df_1d, atr_1d)
    
    # Load 1w data for weekly EMA trend bias
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 2:
        return np.zeros(n)
    
    close_1w = df_1w['close'].values
    ema_50_1w = pd.Series(close_1w).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_50_1w)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    for i in range(50, n):
        # Skip if indicators not ready
        if (np.isnan(r1_aligned[i]) or np.isnan(s1_aligned[i]) or 
            np.isnan(atr_1d_aligned[i]) or np.isnan(ema_50_1w_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        price = prices['close'].iloc[i]
        volume = prices['volume'].iloc[i]
        
        # Volume filter: current volume > 1.3 * 20-period average
        if i >= 20:
            vol_ma = prices['volume'].iloc[i-20:i].mean()
            volume_ok = volume > 1.3 * vol_ma
        else:
            volume_ok = False
        
        # Volatility filter: avoid extremely low volatility periods
        if i >= 20:
            atr_ma = np.nanmean(atr_1d_aligned[i-20:i])
            vol_filter = atr_1d_aligned[i] > 0.5 * atr_ma
        else:
            vol_filter = True
        
        if position == 0:
            # Long breakout: price > R1 with volume and volatility confirmation
            # Only take long if weekly EMA is rising (bullish bias)
            if (price > r1_aligned[i] and 
                ema_50_1w_aligned[i] > ema_50_1w_aligned[i-1] and
                volume_ok and vol_filter):
                signals[i] = 0.25
                position = 1
                entry_price = price
            # Short breakout: price < S1 with volume and volatility confirmation
            # Only take short if weekly EMA is falling (bearish bias)
            elif (price < s1_aligned[i] and 
                  ema_50_1w_aligned[i] < ema_50_1w_aligned[i-1] and
                  volume_ok and vol_filter):
                signals[i] = -0.25
                position = -1
                entry_price = price
        
        elif position == 1:
            # Long stoploss: price < entry_price - 2.0 * ATR
            if price < entry_price - 2.0 * atr_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short stoploss: price > entry_price + 2.0 * ATR
            if price > entry_price + 2.0 * atr_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "1d_Camarilla_R1S1_Breakout_Volume_ATRFilter_v1"
timeframe = "1d"
leverage = 1.0
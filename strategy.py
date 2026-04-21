#!/usr/bin/env python3
"""
Hypothesis: 4h strategy using 1-day Camarilla pivot levels (S1, S2, R1, R2) with 1-day EMA34 trend filter and volume confirmation.
In uptrend (price > 1d EMA34), buy breakouts above R1; in downtrend (price < 1d EMA34), sell breakdowns below S1.
Volume must exceed 1.8x 30-period average to confirm breakout strength. Exit on trend reversal or 2.0x ATR stop.
Designed for 15-25 trades/year (60-100 total over 4 years) to minimize fee drift while capturing major trend moves.
Works in bull markets via R1 breakouts and in bear markets via S1 breakdowns with trend filter.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Load 1d data ONCE before loop for Camarilla pivots and EMA
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 40:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate 1d Camarilla pivot levels
    pivot = (high_1d[-1] + low_1d[-1] + close_1d[-1]) / 3
    range_hl = high_1d[-1] - low_1d[-1]
    r1 = close_1d[-1] + (range_hl * 1.1 / 12)
    s1 = close_1d[-1] - (range_hl * 1.1 / 12)
    r2 = close_1d[-1] + (range_hl * 1.1 / 6)
    s2 = close_1d[-1] - (range_hl * 1.1 / 6)
    r3 = close_1d[-1] + (range_hl * 1.1 / 4)
    s3 = close_1d[-1] - (range_hl * 1.1 / 4)
    
    # Arrays for historical pivots (use previous day's levels)
    r1_hist = np.roll(r1, 1)
    s1_hist = np.roll(s1, 1)
    r2_hist = np.roll(r2, 1)
    s2_hist = np.roll(s2, 1)
    r3_hist = np.roll(r3, 1)
    s3_hist = np.roll(s3, 1)
    r1_hist[0] = r1[0]
    s1_hist[0] = s1[0]
    r2_hist[0] = r2[0]
    s2_hist[0] = s2[0]
    r3_hist[0] = r3[0]
    s3_hist[0] = s3[0]
    
    # Calculate 1d EMA34 for trend filter
    ema_34 = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    
    # Align 1d indicators to 4h timeframe (wait for 1d bar to close)
    r1_aligned = align_htf_to_ltf(prices, df_1d, r1_hist)
    s1_aligned = align_htf_to_ltf(prices, df_1d, s1_hist)
    r2_aligned = align_htf_to_ltf(prices, df_1d, r2_hist)
    s2_aligned = align_htf_to_ltf(prices, df_1d, s2_hist)
    ema_34_aligned = align_htf_to_ltf(prices, df_1d, ema_34)
    
    # Volume confirmation (volume spike > 1.8x 30-period average)
    vol_ma_30 = pd.Series(prices['volume'].values).rolling(window=30, min_periods=30).mean().values
    vol_ratio = prices['volume'].values / vol_ma_30
    
    # ATR for stoploss (20-period)
    tr1 = prices['high'].values - prices['low'].values
    tr2 = np.abs(prices['high'].values - np.roll(prices['close'].values, 1))
    tr3 = np.abs(prices['low'].values - np.roll(prices['close'].values, 1))
    tr2[0] = tr1[0]
    tr3[0] = tr1[0]
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    atr = pd.Series(tr).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(50, n):
        # Skip if indicators not ready
        if (np.isnan(r1_aligned[i]) or np.isnan(s1_aligned[i]) or 
            np.isnan(ema_34_aligned[i]) or np.isnan(vol_ratio[i]) or np.isnan(atr[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        price_close = prices['close'].iloc[i]
        r1_val = r1_aligned[i]
        s1_val = s1_aligned[i]
        ema_trend = ema_34_aligned[i]
        vol_ratio_val = vol_ratio[i]
        atr_val = atr[i]
        
        if position == 0:
            # Enter long: price breaks above S1 (wait, should be R1 for long) - correcting logic
            # Actually: In uptrend (price > EMA), buy breakouts above R1
            # In downtrend (price < EMA), sell breakdowns below S1
            if (price_close > r1_val and 
                price_close > ema_trend and 
                vol_ratio_val > 1.8):
                signals[i] = 0.25
                position = 1
            elif (price_close < s1_val and 
                  price_close < ema_trend and 
                  vol_ratio_val > 1.8):
                signals[i] = -0.25
                position = -1
        
        elif position != 0:
            # Exit: trend reversal OR ATR-based stoploss
            exit_signal = False
            
            # Trend reversal exit
            if position == 1 and price_close < ema_trend:
                exit_signal = True
            elif position == -1 and price_close > ema_trend:
                exit_signal = True
            
            # ATR-based stoploss (2.0x ATR from approximate entry)
            if position == 1:
                # Approximate entry as R1 breakout level
                entry_approx = r1_aligned[i-1] if i > 0 else r1_aligned[i]
                if price_close < entry_approx - 2.0 * atr_val:
                    exit_signal = True
            elif position == -1:
                # Approximate entry as S1 breakdown level
                entry_approx = s1_aligned[i-1] if i > 0 else s1_aligned[i]
                if price_close > entry_approx + 2.0 * atr_val:
                    exit_signal = True
            
            if exit_signal:
                signals[i] = 0.0
                position = 0
            else:
                # Hold position
                signals[i] = 0.25 if position == 1 else -0.25
    
    return signals

name = "4h_Camarilla_S1R1_1dEMA34_Volume_ATR"
timeframe = "4h"
leverage = 1.0
#!/usr/bin/env python3
"""
12h_Camarilla_R1_S1_Breakout_1wTrend_Filter_v1
Hypothesis: Trade Camarilla pivot (R1/S1) breakouts on 12h with 1-week EMA50 trend filter and volume confirmation.
Uses 1-week EMA50 for strong trend adaptation to reduce whipsaws in both bull and bear markets.
Volume spike (2.0x median) confirms participation. Only trade in trending regimes (ADX > 25 on 1d).
ATR-based stoploss (2.5) and profit target (4.0*ATR) manage risk.
Designed for 12-37 trades/year on BTC/ETH/SOL. Works in bull/bear by following 1w EMA50 and filtering chop via ADX.
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
    
    # Get 1-week data for HTF trend
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 50:
        return np.zeros(n)
    
    # Get 1-day data for ADX regime filter and Camarilla calculation
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # 1-week EMA(50) for trend filter
    ema_50_1w = pd.Series(df_1w['close'].values).ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Calculate Camarilla levels from previous 1-day OHLC
    prev_close_1d = df_1d['close'].shift(1).values
    prev_high_1d = df_1d['high'].shift(1).values
    prev_low_1d = df_1d['low'].shift(1).values
    
    camarilla_r1 = prev_close_1d + 1.125 * (prev_high_1d - prev_low_1d)
    camarilla_s1 = prev_close_1d - 1.125 * (prev_high_1d - prev_low_1d)
    
    # Align HTF indicators to 12h timeframe
    ema_50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_50_1w)
    camarilla_r1_aligned = align_htf_to_ltf(prices, df_1d, camarilla_r1)
    camarilla_s1_aligned = align_htf_to_ltf(prices, df_1d, camarilla_s1)
    
    # Volume confirmation: 2.0x median volume (20-period) for signal
    vol_median = pd.Series(volume).rolling(window=20, min_periods=20).median().values
    
    # ATR(14) for volatility-based stops
    tr = np.maximum(high - low, np.maximum(np.abs(high - np.roll(close, 1)), np.abs(low - np.roll(close, 1))))
    tr[0] = high[0] - low[0]  # First period
    atr = pd.Series(tr).ewm(span=14, adjust=False, min_periods=14).mean().values
    
    # ADX(14) on 1d for regime filter - trending when > 25
    plus_dm = np.where((high - np.roll(high, 1)) > (np.roll(low, 1) - low), np.maximum(high - np.roll(high, 1), 0), 0)
    minus_dm = np.where((np.roll(low, 1) - low) > (high - np.roll(high, 1)), np.maximum(np.roll(low, 1) - low, 0), 0)
    plus_dm[0] = 0
    minus_dm[0] = 0
    
    def WilderSmooth(data, period):
        result = np.zeros_like(data)
        if len(data) < period:
            return result
        result[period-1] = np.mean(data[:period])
        for i in range(period, len(data)):
            result[i] = (result[i-1] * (period-1) + data[i]) / period
        return result
    
    period = 14
    if n >= period:
        plus_dm_smooth = WilderSmooth(plus_dm, period)
        minus_dm_smooth = WilderSmooth(minus_dm, period)
        tr_smooth = WilderSmooth(tr, period)
        
        # Avoid division by zero
        plus_di = 100 * plus_dm_smooth / np.where(tr_smooth != 0, tr_smooth, 1)
        minus_di = 100 * minus_dm_smooth / np.where(tr_smooth != 0, tr_smooth, 1)
        dx = 100 * np.abs(plus_di - minus_di) / np.where((plus_di + minus_di) != 0, (plus_di + minus_di), 1)
        adx = WilderSmooth(dx, period)
    else:
        adx = np.full(n, np.nan)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    # Warmup: max of 1w EMA (50), 1d Camarilla (need 2 for shift), volume median (20), ADX (14*2 for smoothing), ATR (14)
    start_idx = max(50, 2, 20, 28, 14)
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(ema_50_1w_aligned[i]) or 
            np.isnan(vol_median[i]) or
            np.isnan(camarilla_r1_aligned[i]) or
            np.isnan(camarilla_s1_aligned[i]) or
            np.isnan(adx[i]) or
            np.isnan(atr[i])):
            # Hold current position
            signals[i] = 0.0 if position == 0 else (0.25 if position == 1 else -0.25)
            continue
        
        ema_50_1w_val = ema_50_1w_aligned[i]
        close_val = close[i]
        volume_val = volume[i]
        vol_median_val = vol_median[i]
        adx_val = adx[i]
        atr_val = atr[i]
        
        if position == 0:
            # Long: break above R1 with volume spike, uptrend (above 1w EMA50), and trending regime
            long_signal = (close_val > camarilla_r1_aligned[i]) and \
                          (volume_val > 2.0 * vol_median_val) and \
                          (close_val > ema_50_1w_val) and \
                          (adx_val > 25)
            
            # Short: break below S1 with volume spike, downtrend (below 1w EMA50), and trending regime
            short_signal = (close_val < camarilla_s1_aligned[i]) and \
                           (volume_val > 2.0 * vol_median_val) and \
                           (close_val < ema_50_1w_val) and \
                           (adx_val > 25)
            
            if long_signal:
                signals[i] = 0.25
                position = 1
                entry_price = close_val
            elif short_signal:
                signals[i] = -0.25
                position = -1
                entry_price = close_val
            else:
                signals[i] = 0.0
        elif position == 1:
            # Hold long
            signals[i] = 0.25
            # Exit conditions
            # 1. Price breaks below S1 (reversal)
            # 2. Trend changes (close < 1w EMA50)
            # 3. Regime changes (ADX < 20)
            # 4. ATR-based stop loss (2.5 * ATR below entry)
            # 5. Profit target (4.0 * ATR above entry)
            if (close_val < camarilla_s1_aligned[i]) or \
               (close_val < ema_50_1w_val) or \
               (adx_val < 20) or \
               (close_val < entry_price - 2.5 * atr_val) or \
               (close_val > entry_price + 4.0 * atr_val):
                signals[i] = 0.0
                position = 0
        elif position == -1:
            # Hold short
            signals[i] = -0.25
            # Exit conditions
            # 1. Price breaks above R1 (reversal)
            # 2. Trend changes (close > 1w EMA50)
            # 3. Regime changes (ADX < 20)
            # 4. ATR-based stop loss (2.5 * ATR above entry)
            # 5. Profit target (4.0 * ATR below entry)
            if (close_val > camarilla_r1_aligned[i]) or \
               (close_val > ema_50_1w_val) or \
               (adx_val < 20) or \
               (close_val > entry_price + 2.5 * atr_val) or \
               (close_val < entry_price - 4.0 * atr_val):
                signals[i] = 0.0
                position = 0
    
    return signals

name = "12h_Camarilla_R1_S1_Breakout_1wTrend_Filter_v1"
timeframe = "12h"
leverage = 1.0
#!/usr/bin/env python3
"""
Hypothesis: 4h strategy using 1d Camarilla pivot levels (R1/S1) with 1d EMA34 trend filter and volume confirmation.
In uptrend (price > 1d EMA34), buy touches of 1d Camarilla S1 level with rejection; in downtrend (price < 1d EMA34), sell touches of 1d Camarilla R1 level with rejection.
Volume must exceed 1.5x 20-period average to confirm rejection strength. Exit on trend reversal or 2x ATR stop.
Designed for 20-50 trades/year (80-200 total over 4 years) to minimize fee fade while capturing mean-reversion bounces at institutional levels.
Works in bull markets via S1 bounces and in bear markets via R1 rejections with trend filter.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Load 1d data ONCE before loop for Camarilla and EMA
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 34:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate 1d Camarilla levels (based on previous day)
    # Camarilla: H4 = C + 1.5*(H-L), L4 = C - 1.5*(H-L)
    # R1 = C + (H-L)*1.1/12, S1 = C - (H-L)*1.1/12
    prev_close = np.roll(close_1d, 1)
    prev_high = np.roll(high_1d, 1)
    prev_low = np.roll(low_1d, 1)
    # First day: use same values
    prev_close[0] = close_1d[0]
    prev_high[0] = high_1d[0]
    prev_low[0] = low_1d[0]
    
    cam_r1 = prev_close + (prev_high - prev_low) * 1.1 / 12
    cam_s1 = prev_close - (prev_high - prev_low) * 1.1 / 12
    
    # Calculate 1d EMA34 for trend filter
    ema_34 = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    
    # Align 1d indicators to 4h timeframe (wait for 1d bar to close)
    cam_r1_aligned = align_htf_to_ltf(prices, df_1d, cam_r1)
    cam_s1_aligned = align_htf_to_ltf(prices, df_1d, cam_s1)
    ema_34_aligned = align_htf_to_ltf(prices, df_1d, ema_34)
    
    # Volume confirmation (volume spike > 1.5x 20-period average)
    vol_ma_20 = pd.Series(prices['volume'].values).rolling(window=20, min_periods=20).mean().values
    vol_ratio = prices['volume'].values / vol_ma_20
    
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
    
    for i in range(100, n):
        # Skip if indicators not ready
        if (np.isnan(cam_r1_aligned[i]) or np.isnan(cam_s1_aligned[i]) or 
            np.isnan(ema_34_aligned[i]) or np.isnan(vol_ratio[i]) or np.isnan(atr[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        price_close = prices['close'].iloc[i]
        price_low = prices['low'].iloc[i]
        price_high = prices['high'].iloc[i]
        r1_level = cam_r1_aligned[i]
        s1_level = cam_s1_aligned[i]
        ema_trend = ema_34_aligned[i]
        vol_ratio_val = vol_ratio[i]
        atr_val = atr[i]
        
        if position == 0:
            # Enter long: price touches S1 and shows rejection (close > open) in uptrend
            if (price_low <= s1_level * 1.002 and  # Allow 0.2% tolerance for touch
                price_close > prices['open'].iloc[i] and  # Bullish candle
                price_close > ema_trend and 
                vol_ratio_val > 1.5):
                signals[i] = 0.25
                position = 1
            # Enter short: price touches R1 and shows rejection (close < open) in downtrend
            elif (price_high >= r1_level * 0.998 and  # Allow 0.2% tolerance for touch
                  price_close < prices['open'].iloc[i] and  # Bearish candle
                  price_close < ema_trend and 
                  vol_ratio_val > 1.5):
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
            
            # ATR-based stoploss (2x ATR from entry level)
            if position == 1:
                entry_approx = s1_level  # Entered near S1
                if price_close < entry_approx - 2.0 * atr_val:
                    exit_signal = True
            elif position == -1:
                entry_approx = r1_level  # Entered near R1
                if price_close > entry_approx + 2.0 * atr_val:
                    exit_signal = True
            
            if exit_signal:
                signals[i] = 0.0
                position = 0
            else:
                # Hold position
                signals[i] = 0.25 if position == 1 else -0.25
    
    return signals

name = "4h_Camarilla_R1_S1_Rejection_1dEMA34_Volume_ATR"
timeframe = "4h"
leverage = 1.0
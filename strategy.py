#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1h strategy using 4h Camarilla R3/S3 breakout with 1d EMA200 trend filter and UTC 08-20 session.
# Uses 4h for signal direction (Camarilla breakouts), 1d EMA200 for trend alignment, and 1h for precise entry timing.
# Session filter reduces noise trades. Discrete position sizing (0.20) minimizes fee churn.
# Target: 60-150 total trades over 4 years (15-37/year) to avoid fee drag while maintaining statistical significance.
# Works in both bull/bear: trend filter adapts to market regime, session filter avoids low-liquidity hours.

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # === 4h data (HTF for signal direction) ===
    df_4h = get_htf_data(prices, '4h')
    high_4h = df_4h['high'].values
    low_4h = df_4h['low'].values
    close_4h = df_4h['close'].values
    
    # === 1d data (HTF for trend filter) ===
    df_1d = get_htf_data(prices, '1d')
    close_1d = df_1d['close'].values
    
    # === 1d EMA200 for trend filter ===
    ema200_1d = pd.Series(close_1d).ewm(span=200, adjust=False, min_periods=200).mean().values
    ema200_1d_aligned = align_htf_to_ltf(prices, df_1d, ema200_1d)
    
    # === 4h Camarilla pivot levels (R3, S3) ===
    # Camarilla formula: R3 = C + (H-L)*1.1/4, S3 = C - (H-L)*1.1/4
    camarilla_r3_4h = close_4h + (high_4h - low_4h) * 1.1 / 4
    camarilla_s3_4h = close_4h - (high_4h - low_4h) * 1.1 / 4
    
    # Align 4h Camarilla levels to 1h timeframe (wait for 4h bar close)
    camarilla_r3_4h_aligned = align_htf_to_ltf(prices, df_4h, camarilla_r3_4h)
    camarilla_s3_4h_aligned = align_htf_to_ltf(prices, df_4h, camarilla_s3_4h)
    
    # === Session filter: UTC 08-20 ===
    hours = prices.index.hour  # open_time is already datetime64[ms]
    in_session = (hours >= 8) & (hours <= 20)
    
    signals = np.zeros(n)
    
    # Warmup
    warmup = 200
    
    # Track position and entry price for stoploss
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    for i in range(warmup, n):
        # Skip if any data is NaN
        if (np.isnan(ema200_1d_aligned[i]) or 
            np.isnan(camarilla_r3_4h_aligned[i]) or
            np.isnan(camarilla_s3_4h_aligned[i])):
            signals[i] = 0.0
            position = 0
            continue
        
        # Apply session filter
        if not in_session[i]:
            # Force flat outside session
            if position == 1:
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
                continue
            elif position == -1:
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
                continue
            else:
                signals[i] = 0.0
                continue
        
        price = close[i]
        r3 = camarilla_r3_4h_aligned[i]
        s3 = camarilla_s3_4h_aligned[i]
        ema200 = ema200_1d_aligned[i]
        
        # === STOPLOSS LOGIC (ATR-based) ===
        if position == 1:  # Long position
            atr_4h = np.abs(high_4h - low_4h)
            atr_ma = pd.Series(atr_4h).rolling(window=14, min_periods=14).mean().values
            atr_aligned = align_htf_to_ltf(prices, df_4h, atr_ma)
            atr_val = atr_aligned[i]
            if price < entry_price - 2.0 * atr_val:
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
                continue
        
        elif position == -1:  # Short position
            atr_4h = np.abs(high_4h - low_4h)
            atr_ma = pd.Series(atr_4h).rolling(window=14, min_periods=14).mean().values
            atr_aligned = align_htf_to_ltf(prices, df_4h, atr_ma)
            atr_val = atr_aligned[i]
            if price > entry_price + 2.0 * atr_val:
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
                continue
        
        # === EXIT LOGIC ===
        if position == 1:  # Long position
            # Exit when price reaches 4h R4 or shows weakness below R3
            camarilla_r4_4h = close_4h + (high_4h - low_4h) * 1.1/2
            camarilla_r4_4h_aligned = align_htf_to_ltf(prices, df_4h, camarilla_r4_4h)
            if not np.isnan(camarilla_r4_4h_aligned[i]) and price >= camarilla_r4_4h_aligned[i]:
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
                continue
            # Exit if price falls back below R3 (failed breakout)
            if price < r3:
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
                continue
        
        elif position == -1:  # Short position
            # Exit when price reaches 4h S4 or shows strength above S3
            camarilla_s4_4h = close_4h - (high_4h - low_4h) * 1.1/2
            camarilla_s4_4h_aligned = align_htf_to_ltf(prices, df_4h, camarilla_s4_4h)
            if not np.isnan(camarilla_s4_4h_aligned[i]) and price <= camarilla_s4_4h_aligned[i]:
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
                continue
            # Exit if price rises back above S3 (failed breakdown)
            if price > s3:
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
                continue
        
        # === ENTRY LOGIC (only when flat) ===
        if position == 0:
            # Require trend alignment via 1d EMA200
            # Go long when price breaks above 4h R3 and above 1d EMA200 (bullish alignment)
            if price > r3 and price > ema200:
                signals[i] = 0.20
                position = 1
                entry_price = price
                continue
            # Go short when price breaks below 4h S3 and below 1d EMA200 (bearish alignment)
            elif price < s3 and price < ema200:
                signals[i] = -0.20
                position = -1
                entry_price = price
                continue
        
        # Hold current position
        if position == 1:
            signals[i] = 0.20
        elif position == -1:
            signals[i] = -0.20
        else:
            signals[i] = 0.0
    
    return signals

name = "1h_Camarilla_R3S3_Breakout_1dEMA200_SessionFilter"
timeframe = "1h"
leverage = 1.0
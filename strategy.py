#!/usr/bin/env python3
"""
4h_Camarilla_Reversal_Scalp
Hypothesis: Camarilla pivot levels (H4/L4) act as strong intraday support/resistance. When price reaches H4 or L4 with confirmation from volume spike and RSI extreme, it signals a high-probability reversal. The 12h EMA50 filters the trend direction to avoid counter-trend trades in strong trends. Designed for 4h timeframe to capture reversals in both ranging and trending markets, with low trade frequency to minimize fee drag.
"""

name = "4h_Camarilla_Reversal_Scalp"
timeframe = "4h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Get 12h data for EMA trend filter
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 20:
        return np.zeros(n)
    
    # Get 1d data for Camarilla pivot calculation
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    # 4h OHLCV
    close_4h = prices['close'].values
    high_4h = prices['high'].values
    low_4h = prices['low'].values
    volume_4h = prices['volume'].values
    
    # --- 12h EMA50 for trend filter ---
    close_12h = df_12h['close'].values
    ema_50_12h = pd.Series(close_12h).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_12h_aligned = align_htf_to_ltf(prices, df_12h, ema_50_12h)
    
    # --- 1d Camarilla pivot levels (H4, L4, H3, L3) ---
    # Calculated from previous day's OHLC
    prev_day_high = np.roll(df_1d['high'].values, 1)
    prev_day_low = np.roll(df_1d['low'].values, 1)
    prev_day_close = np.roll(df_1d['close'].values, 1)
    # First day: use same day's values
    prev_day_high[0] = df_1d['high'].values[0]
    prev_day_low[0] = df_1d['low'].values[0]
    prev_day_close[0] = df_1d['close'].values[0]
    
    # Camarilla formulas
    camarilla_h4 = prev_day_close + 1.1 * (prev_day_high - prev_day_low) / 2
    camarilla_l4 = prev_day_close - 1.1 * (prev_day_high - prev_day_low) / 2
    camarilla_h3 = prev_day_close + 1.1 * (prev_day_high - prev_day_low) / 4
    camarilla_l3 = prev_day_close - 1.1 * (prev_day_high - prev_day_low) / 4
    
    # Align daily levels to 4h
    h4_4h = align_htf_to_ltf(prices, df_1d, camarilla_h4)
    l4_4h = align_htf_to_ltf(prices, df_1d, camarilla_l4)
    h3_4h = align_htf_to_ltf(prices, df_1d, camarilla_h3)
    l3_4h = align_htf_to_ltf(prices, df_1d, camarilla_l3)
    
    # --- 4h RSI(14) for overbought/oversold ---
    delta = np.diff(close_4h, prepend=close_4h[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    avg_gain = pd.Series(gain).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    avg_loss = pd.Series(loss).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    rs = avg_gain / (avg_loss + 1e-10)
    rsi = 100 - (100 / (1 + rs))
    
    # --- 4h Volume average for confirmation ---
    vol_avg_4h = pd.Series(volume_4h).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    # Start after warmup period
    start_idx = 50  # for EMA and RSI
    
    for i in range(start_idx, n):
        # Skip if any critical values are NaN
        if (np.isnan(ema_50_12h_aligned[i]) or np.isnan(h4_4h[i]) or 
            np.isnan(l4_4h[i]) or np.isnan(rsi[i]) or np.isnan(vol_avg_4h[i])):
            if position != 0:
                # Simple stoploss: 1.5x ATR from entry
                atr_est = np.abs(high_4h[i] - low_4h[i])
                if position == 1 and close_4h[i] <= entry_price - 1.5 * atr_est:
                    signals[i] = 0.0
                    position = 0
                elif position == -1 and close_4h[i] >= entry_price + 1.5 * atr_est:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = 0.25 if position == 1 else -0.25
            continue
        
        # Volume confirmation: current volume > 1.5x 4h average
        vol_confirm = volume_4h[i] > 1.5 * vol_avg_4h[i]
        
        # Trend filter: only trade in direction of 12h EMA50
        uptrend = close_4h[i] > ema_50_12h_aligned[i]
        downtrend = close_4h[i] < ema_50_12h_aligned[i]
        
        if position == 0:
            # Look for reversal entries at Camarilla levels
            # Long setup: price at L4 with RSI oversold and volume spike
            if (close_4h[i] <= l4_4h[i] * 1.001 and  # allow small slippage
                rsi[i] < 30 and 
                vol_confirm and 
                downtrend):  # only long in downtrend (mean reversion)
                signals[i] = 0.25
                position = 1
                entry_price = close_4h[i]
            # Short setup: price at H4 with RSI overbought and volume spike
            elif (close_4h[i] >= h4_4h[i] * 0.999 and  # allow small slippage
                  rsi[i] > 70 and 
                  vol_confirm and 
                  uptrend):  # only short in uptrend (mean reversion)
                signals[i] = -0.25
                position = -1
                entry_price = close_4h[i]
            # Alternative: H3/L3 breakouts with trend (for stronger moves)
            elif vol_confirm:
                # Long breakout above H3 in uptrend
                if (close_4h[i] > h3_4h[i] and 
                    uptrend and 
                    rsi[i] > 50):  # bullish momentum
                    signals[i] = 0.25
                    position = 1
                    entry_price = close_4h[i]
                # Short breakdown below L3 in downtrend
                elif (close_4h[i] < l3_4h[i] and 
                      downtrend and 
                      rsi[i] < 50):  # bearish momentum
                    signals[i] = -0.25
                    position = -1
                    entry_price = close_4h[i]
        else:
            # Manage existing position
            if position == 1:
                # Long position: exit at H3 (profit) or L4 (stop)
                if close_4h[i] >= h3_4h[i]:
                    signals[i] = 0.0
                    position = 0
                elif close_4h[i] <= l4_4h[i]:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = 0.25
            elif position == -1:
                # Short position: exit at L3 (profit) or H4 (stop)
                if close_4h[i] <= l3_4h[i]:
                    signals[i] = 0.0
                    position = 0
                elif close_4h[i] >= h4_4h[i]:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = -0.25
    
    return signals
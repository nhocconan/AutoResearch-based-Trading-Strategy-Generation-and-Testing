#!/usr/bin/env python3
# 1d_1w_vwap_trend_v1
# Strategy: Daily VWAP position with weekly trend filter and volume confirmation
# Timeframe: 1d
# Leverage: 1.0
# Hypothesis: Price above VWAP indicates bullish momentum, below VWAP indicates bearish momentum.
# Weekly trend filter (close above/below 20-week EMA) ensures trades align with higher timeframe momentum.
# Volume confirmation filters weak signals. Works in bull by buying pullbacks to VWAP in uptrend,
# and in bear by selling bounces to VWAP in downtrend.

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "1d_1w_vwap_trend_v1"
timeframe = "1d"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Price arrays
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Load weekly data ONCE before loop
    df_1w = get_htf_data(prices, '1w')
    
    if len(df_1w) < 30:
        return np.zeros(n)
    
    # Weekly EMA for trend filter (20-period)
    close_1w = df_1w['close'].values
    ema_20_1w = pd.Series(close_1w).ewm(span=20, adjust=False, min_periods=20).mean().values
    weekly_trend = close_1w > ema_20_1w  # Uptrend when close > 20-week EMA
    weekly_trend_aligned = align_htf_to_ltf(prices, df_1w, weekly_trend)
    
    # Daily VWAP calculation
    typical_price = (high + low + close) / 3.0
    pv = typical_price * volume
    cum_pv = np.nancumsum(pv)
    cum_vol = np.nancumsum(volume)
    vwap = cum_pv / (cum_vol + 1e-10)
    
    # Volume average (20-period) for confirmation
    vol_avg = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_spike = volume > (1.5 * vol_avg)  # Volume spike filter
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(20, n):  # Start after VWAP warmup
        # Skip if any required data is invalid
        if (np.isnan(vwap[i]) or np.isnan(weekly_trend_aligned[i]) or np.isnan(vol_avg[i])):
            signals[i] = 0.0 if position == 0 else (0.25 if position == 1 else -0.25)
            continue
        
        # Current values
        price_now = close[i]
        vwap_now = vwap[i]
        trend_now = weekly_trend_aligned[i]
        
        # Entry conditions
        long_entry = price_now > vwap_now and trend_now and vol_spike[i]
        short_entry = price_now < vwap_now and not trend_now and vol_spike[i]
        
        # Exit conditions: price crosses VWAP or opposite signal
        exit_long = position == 1 and price_now < vwap_now
        exit_short = position == -1 and price_now > vwap_now
        
        # Trading logic
        if long_entry and position != 1:
            position = 1
            signals[i] = 0.25
        elif short_entry and position != -1:
            position = -1
            signals[i] = -0.25
        elif position == 1 and exit_long:
            position = 0
            signals[i] = 0.0
        elif position == -1 and exit_short:
            position = 0
            signals[i] = 0.0
        else:
            signals[i] = 0.25 if position == 1 else (-0.25 if position == -1 else 0.0)
    
    return signals
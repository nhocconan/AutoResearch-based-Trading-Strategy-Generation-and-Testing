#!/usr/bin/env python3
# 1d_1w_camarilla_high_volume_v1
# Strategy: Daily Camarilla pivot levels with weekly trend filter and volume confirmation
# Timeframe: 1d
# Leverage: 1.0
# Hypothesis: Price tends to reverse at Camarilla pivot levels (H3/L3) during ranging markets,
# but continues in the direction of the weekly trend during strong trends. Volume confirms
# institutional participation. Works in bull markets by buying dips in uptrends and in bear
# markets by selling rallies in downtrends. Low trade frequency avoids fee drag.

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "1d_1w_camarilla_high_volume_v1"
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
    
    # Load higher timeframe data ONCE before loop
    df_1w = get_htf_data(prices, '1w')
    
    if len(df_1w) < 50:
        return np.zeros(n)
    
    # Weekly EMA200 for trend filter
    close_1w = df_1w['close'].values
    ema_200_1w = pd.Series(close_1w).ewm(span=200, adjust=False, min_periods=200).mean().values
    
    # Align weekly data to daily timeframe (wait for weekly close)
    ema_200_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_200_1w)
    
    # Daily OHLC for Camarilla calculation
    # Calculate Camarilla levels using previous day's OHLC
    # H3 = Close + (High - Low) * 1.1/2
    # L3 = Close - (High - Low) * 1.1/2
    prev_high = np.roll(high, 1)
    prev_low = np.roll(low, 1)
    prev_close = np.roll(close, 1)
    
    # First day has no previous data
    prev_high[0] = high[0]
    prev_low[0] = low[0]
    prev_close[0] = close[0]
    
    rang = prev_high - prev_low
    H3 = prev_close + rang * 1.1 / 2
    L3 = prev_close - rang * 1.1 / 2
    
    # Volume average (20-period) for confirmation
    vol_avg = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_spike = volume > (2.0 * vol_avg)  # Require strong volume spike
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(50, n):
        # Skip if any required data is invalid
        if (np.isnan(ema_200_1w_aligned[i]) or np.isnan(H3[i]) or 
            np.isnan(L3[i]) or np.isnan(vol_avg[i])):
            signals[i] = 0.0 if position == 0 else (0.25 if position == 1 else -0.25)
            continue
        
        price_close = close[i]
        
        # Trend filter: price above/below weekly EMA200
        uptrend_1w = price_close > ema_200_1w_aligned[i]
        downtrend_1w = price_close < ema_200_1w_aligned[i]
        
        # Mean reversion at Camarilla levels with volume confirmation
        # Long at L3 in uptrend (buy the dip)
        long_signal = (price_close <= L3[i]) and vol_spike[i] and uptrend_1w
        # Short at H3 in downtrend (sell the rally)
        short_signal = (price_close >= H3[i]) and vol_spike[i] and downtrend_1w
        
        # Exit when price returns to previous day's close (mean reversion complete)
        exit_long = position == 1 and (price_close >= prev_close[i])
        exit_short = position == -1 and (price_close <= prev_close[i])
        
        # Trading logic
        if long_signal and position != 1:
            position = 1
            signals[i] = 0.25
        elif short_signal and position != -1:
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
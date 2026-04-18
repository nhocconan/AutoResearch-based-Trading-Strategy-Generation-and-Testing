#!/usr/bin/env python3
"""
1d_Weekly_Pivot_Reversal
Hypothesis: On daily timeframe, price reverses at weekly pivot support/resistance levels.
In weekly downtrend, long when price touches weekly S1 with bullish engulfing candle.
In weekly uptrend, short when price touches weekly R1 with bearish engulfing candle.
Uses weekly trend filter to avoid counter-trend trades. Designed for low frequency (5-15 trades/year)
with high win rate in ranging and trending markets. Works in both bull and bear regimes by
following the weekly trend direction while fading intraday extremes at pivot levels.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    open_price = prices['open'].values
    volume = prices['volume'].values
    
    # Get weekly data ONCE before loop
    df_weekly = get_htf_data(prices, '1w')
    if len(df_weekly) == 0:
        return np.zeros(n)
    
    # Calculate weekly pivot points (using previous week's OHLC)
    weekly_high = df_weekly['high'].values
    weekly_low = df_weekly['low'].values
    weekly_close = df_weekly['close'].values
    weekly_open = df_weekly['open'].values
    
    # Pivot = (H + L + C) / 3
    weekly_pivot = (weekly_high + weekly_low + weekly_close) / 3
    # Support 1 = (2 * Pivot) - High
    weekly_s1 = (2 * weekly_pivot) - weekly_high
    # Resistance 1 = (2 * Pivot) - Low
    weekly_r1 = (2 * weekly_pivot) - weekly_low
    
    # Weekly trend: EMA9 vs EMA21 on weekly close
    weekly_ema9 = pd.Series(weekly_close).ewm(span=9, adjust=False).values
    weekly_ema21 = pd.Series(weekly_close).ewm(span=21, adjust=False).values
    weekly_uptrend = weekly_ema9 > weekly_ema21
    
    # Align weekly data to daily
    weekly_pivot_daily = align_htf_to_ltf(prices, df_weekly, weekly_pivot)
    weekly_s1_daily = align_htf_to_ltf(prices, df_weekly, weekly_s1)
    weekly_r1_daily = align_htf_to_ltf(prices, df_weekly, weekly_r1)
    weekly_uptrend_daily = align_htf_to_ltf(prices, df_weekly, weekly_uptrend.astype(float))
    
    # Daily volume average (20-period)
    vol_ma = np.full(n, np.nan)
    for i in range(20, n):
        vol_ma[i] = np.mean(volume[i-20:i])
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 30  # Warmup for weekly alignment and volume
    
    for i in range(start_idx, n):
        if (np.isnan(weekly_pivot_daily[i]) or np.isnan(weekly_s1_daily[i]) or 
            np.isnan(weekly_r1_daily[i]) or np.isnan(weekly_uptrend_daily[i]) or
            np.isnan(vol_ma[i])):
            signals[i] = 0.0
            continue
        
        # Bullish engulfing: current green candle engulfs previous red candle
        bullish_engulf = (close[i] > open_price[i]) and (open_price[i-1] > close[i-1]) and \
                         (close[i] >= open_price[i-1]) and (open_price[i] <= close[i-1])
        # Bearish engulfing: current red candle engulfs previous green candle
        bearish_engulf = (close[i] < open_price[i]) and (open_price[i-1] < close[i-1]) and \
                         (open_price[i] >= close[i-1]) and (close[i] <= open_price[i-1])
        
        if position == 0:
            # Long: weekly uptrend + price at S1 + bullish engulfing + volume confirmation
            if (weekly_uptrend_daily[i] > 0.5 and 
                low[i] <= weekly_s1_daily[i] * 1.005 and  # Allow 0.5% slippage
                bullish_engulf and 
                volume[i] > vol_ma[i] * 1.5):
                signals[i] = 0.25
                position = 1
            # Short: weekly downtrend + price at R1 + bearish engulfing + volume confirmation
            elif (weekly_uptrend_daily[i] < 0.5 and 
                  high[i] >= weekly_r1_daily[i] * 0.995 and  # Allow 0.5% slippage
                  bearish_engulf and 
                  volume[i] > vol_ma[i] * 1.5):
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit: weekly trend turns down OR price reaches pivot
            if (weekly_uptrend_daily[i] < 0.5 or 
                high[i] >= weekly_pivot_daily[i] * 0.995):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit: weekly trend turns up OR price reaches pivot
            if (weekly_uptrend_daily[i] > 0.5 or 
                low[i] <= weekly_pivot_daily[i] * 1.005):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "1d_Weekly_Pivot_Reversal"
timeframe = "1d"
leverage = 1.0
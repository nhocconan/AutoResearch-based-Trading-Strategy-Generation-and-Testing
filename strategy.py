#!/usr/bin/env python3
"""
Hypothesis: 6h Camarilla R3/S3 mean reversion with 1d trend filter and volume confirmation.
Long when price touches S3 AND daily close > daily EMA50 AND volume > 1.5x average.
Short when price touches R3 AND daily close < daily EMA50 AND volume > 1.5x average.
Exit when price crosses the daily VWAP (mean reversion completion).
Uses discrete position sizing (0.25) to minimize fee churn. Targets 20-40 trades/year per symbol.
Camarilla levels provide precise intraday support/resistance, daily trend filter avoids counter-trend trades.
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
    
    # Load 1d data for trend filter and VWAP - ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    volume_1d = df_1d['volume'].values
    
    # Calculate EMA50 on 1d data
    ema50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Calculate VWAP on 1d data (typical price * volume)
    typical_price_1d = (high_1d + low_1d + close_1d) / 3.0
    vwap_1d = (pd.Series(typical_price_1d * volume_1d).cumsum() / 
               pd.Series(volume_1d).cumsum()).values
    
    # Align 1d indicators to 6h timeframe
    ema50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema50_1d)
    vwap_1d_aligned = align_htf_to_ltf(prices, df_1d, vwap_1d)
    
    # Calculate Camarilla levels from previous 1d bar
    # Camarilla uses previous day's high, low, close
    prev_close_1d = np.roll(close_1d, 1)
    prev_high_1d = np.roll(high_1d, 1)
    prev_low_1d = np.roll(low_1d, 1)
    prev_close_1d[0] = close_1d[0]  # first bar uses current bar
    prev_high_1d[0] = high_1d[0]
    prev_low_1d[0] = low_1d[0]
    
    # Camarilla R3, S3 levels
    rangep = prev_high_1d - prev_low_1d
    camarilla_r3 = prev_close_1d + rangep * 1.1 / 4
    camarilla_s3 = prev_close_1d - rangep * 1.1 / 4
    
    # Align Camarilla levels to 6h timeframe
    camarilla_r3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_r3)
    camarilla_s3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_s3)
    
    # Volume average (20-period) on primary timeframe
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(100, n):
        # Skip if data not ready
        if (np.isnan(ema50_1d_aligned[i]) or np.isnan(vwap_1d_aligned[i]) or
            np.isnan(camarilla_r3_aligned[i]) or np.isnan(camarilla_s3_aligned[i]) or
            np.isnan(vol_ma[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        daily_trend_up = close[i] > ema50_1d_aligned[i]  # using 6h close vs daily EMA
        daily_trend_down = close[i] < ema50_1d_aligned[i]
        
        vol_current = volume[i]
        vol_ma_val = vol_ma[i]
        
        if position == 0:
            # Long: Price touches S3 AND daily uptrend AND volume confirmation
            if (low[i] <= camarilla_s3_aligned[i] and daily_trend_up and 
                vol_current > 1.5 * vol_ma_val):
                signals[i] = 0.25
                position = 1
            # Short: Price touches R3 AND daily downtrend AND volume confirmation
            elif (high[i] >= camarilla_r3_aligned[i] and daily_trend_down and 
                  vol_current > 1.5 * vol_ma_val):
                signals[i] = -0.25
                position = -1
        else:
            # Exit conditions
            exit_signal = False
            
            if position == 1:
                # Exit long: Price crosses above daily VWAP (mean reversion complete)
                if close[i] > vwap_1d_aligned[i]:
                    exit_signal = True
            else:  # position == -1
                # Exit short: Price crosses below daily VWAP (mean reversion complete)
                if close[i] < vwap_1d_aligned[i]:
                    exit_signal = True
            
            if exit_signal:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25 if position == 1 else -0.25
    
    return signals

name = "6H_Camarilla_R3S3_1dEMA50_VWAP"
timeframe = "6h"
leverage = 1.0
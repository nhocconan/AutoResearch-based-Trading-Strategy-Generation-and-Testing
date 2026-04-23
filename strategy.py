#!/usr/bin/env python3
"""
Hypothesis: 1h Camarilla R3/S3 mean reversion with 4h trend filter and volume confirmation.
Long when price touches S3 AND 4h close > 4h EMA50 AND volume > 1.8x average.
Short when price touches R3 AND 4h close < 4h EMA50 AND volume > 1.8x average.
Exit when price crosses the 4h VWAP (mean reversion completion).
Uses discrete position sizing (0.20) to minimize fee churn. Targets 15-37 trades/year per symbol.
Camarilla levels provide precise intraday support/resistance, 4h trend filter avoids counter-trend trades.
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
    
    # Load 4h data for trend filter and VWAP - ONCE before loop
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 50:
        return np.zeros(n)
    
    close_4h = df_4h['close'].values
    high_4h = df_4h['high'].values
    low_4h = df_4h['low'].values
    volume_4h = df_4h['volume'].values
    
    # Calculate EMA50 on 4h data
    ema50_4h = pd.Series(close_4h).ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Calculate VWAP on 4h data (typical price * volume)
    typical_price_4h = (high_4h + low_4h + close_4h) / 3.0
    vwap_4h = (pd.Series(typical_price_4h * volume_4h).cumsum() / 
               pd.Series(volume_4h).cumsum()).values
    
    # Align 4h indicators to 1h timeframe
    ema50_4h_aligned = align_htf_to_ltf(prices, df_4h, ema50_4h)
    vwap_4h_aligned = align_htf_to_ltf(prices, df_4h, vwap_4h)
    
    # Calculate Camarilla levels from previous 4h bar
    # Camarilla uses previous bar's high, low, close
    prev_close_4h = np.roll(close_4h, 1)
    prev_high_4h = np.roll(high_4h, 1)
    prev_low_4h = np.roll(low_4h, 1)
    prev_close_4h[0] = close_4h[0]  # first bar uses current bar
    prev_high_4h[0] = high_4h[0]
    prev_low_4h[0] = low_4h[0]
    
    # Camarilla R3, S3 levels
    rangep = prev_high_4h - prev_low_4h
    camarilla_r3 = prev_close_4h + rangep * 1.1 / 4
    camarilla_s3 = prev_close_4h - rangep * 1.1 / 4
    
    # Align Camarilla levels to 1h timeframe
    camarilla_r3_aligned = align_htf_to_ltf(prices, df_4h, camarilla_r3)
    camarilla_s3_aligned = align_htf_to_ltf(prices, df_4h, camarilla_s3)
    
    # Volume average (20-period) on primary timeframe
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(100, n):
        # Skip if data not ready
        if (np.isnan(ema50_4h_aligned[i]) or np.isnan(vwap_4h_aligned[i]) or
            np.isnan(camarilla_r3_aligned[i]) or np.isnan(camarilla_s3_aligned[i]) or
            np.isnan(vol_ma[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Session filter: 08-20 UTC
        hour = pd.Timestamp(prices['open_time'].iloc[i]).hour
        if hour < 8 or hour > 20:
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # 4h trend filter
        daily_trend_up = close[i] > ema50_4h_aligned[i]  # using 1h close vs 4h EMA
        daily_trend_down = close[i] < ema50_4h_aligned[i]
        
        vol_current = volume[i]
        vol_ma_val = vol_ma[i]
        
        if position == 0:
            # Long: Price touches S3 AND 4h uptrend AND volume confirmation
            if (low[i] <= camarilla_s3_aligned[i] and daily_trend_up and 
                vol_current > 1.8 * vol_ma_val):
                signals[i] = 0.20
                position = 1
            # Short: Price touches R3 AND 4h downtrend AND volume confirmation
            elif (high[i] >= camarilla_r3_aligned[i] and daily_trend_down and 
                  vol_current > 1.8 * vol_ma_val):
                signals[i] = -0.20
                position = -1
        else:
            # Exit conditions
            exit_signal = False
            
            if position == 1:
                # Exit long: Price crosses above 4h VWAP (mean reversion complete)
                if close[i] > vwap_4h_aligned[i]:
                    exit_signal = True
            else:  # position == -1
                # Exit short: Price crosses below 4h VWAP (mean reversion complete)
                if close[i] < vwap_4h_aligned[i]:
                    exit_signal = True
            
            if exit_signal:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.20 if position == 1 else -0.20
    
    return signals

name = "1H_Camarilla_R3S3_4hEMA50_VWAP"
timeframe = "1h"
leverage = 1.0
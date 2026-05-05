#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1h Camarilla R3/S3 breakout with 4h EMA50 trend filter and volume spike confirmation
# Long when price breaks above Camarilla R3 AND price > EMA50(4h) AND volume > 2x 20-period average
# Short when price breaks below Camarilla S3 AND price < EMA50(4h) AND volume > 2x 20-period average
# Exit when price retests Camarilla pivot point (PP) OR trend flips (price crosses EMA50(4h))
# Uses discrete sizing (0.20) to limit fee drag. Target: 15-35 trades/year per symbol.
# Camarilla levels provide precise intraday support/resistance, 4h EMA50 filters counter-trend trades,
# volume spike confirms institutional participation. Works in both bull (buying strength) and bear (selling pressure).

name = "1h_Camarilla_R3S3_Breakout_4hEMA50_Trend_VolumeSpike"
timeframe = "1h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Get 4h data ONCE before loop for EMA50 trend filter
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 2:
        return np.zeros(n)
    
    # Calculate EMA50 on 4h close for trend filter
    close_4h = df_4h['close'].values
    ema_50 = pd.Series(close_4h).ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Align 4h EMA50 to 1h timeframe
    ema_50_aligned = align_htf_to_ltf(prices, df_4h, ema_50)
    
    # Calculate Camarilla levels using previous day's OHLC
    # Need daily data for pivot calculation
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    # Calculate daily OHLC for Camarilla
    daily_high = df_1d['high'].values
    daily_low = df_1d['low'].values
    daily_close = df_1d['close'].values
    
    # Camarilla levels: PP = (H+L+C)/3, R3 = PP + (H-L)*1.1/2, S3 = PP - (H-L)*1.1/2
    PP = (daily_high + daily_low + daily_close) / 3
    R3 = PP + (daily_high - daily_low) * 1.1 / 2
    S3 = PP - (daily_high - daily_low) * 1.1 / 2
    
    # Align Camarilla levels to 1h timeframe (use previous day's levels)
    PP_aligned = align_htf_to_ltf(prices, df_1d, PP, additional_delay_bars=1)
    R3_aligned = align_htf_to_ltf(prices, df_1d, R3, additional_delay_bars=1)
    S3_aligned = align_htf_to_ltf(prices, df_1d, S3, additional_delay_bars=1)
    
    # Volume confirmation: volume > 2x 20-period average (spike filter)
    if len(volume) >= 20:
        vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
        volume_filter = volume > (2.0 * vol_ma_20)
    else:
        volume_filter = np.zeros(n, dtype=bool)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(50, n):
        # Skip if any value is NaN
        if (np.isnan(ema_50_aligned[i]) or 
            np.isnan(R3_aligned[i]) or 
            np.isnan(S3_aligned[i]) or 
            np.isnan(PP_aligned[i]) or 
            np.isnan(volume_filter[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long conditions: price breaks above R3 AND price > EMA50(4h) AND volume spike
            if (close[i] > R3_aligned[i] and 
                close[i-1] <= R3_aligned[i-1] and  # ensure breakout just happened
                close[i] > ema_50_aligned[i] and 
                volume_filter[i]):
                signals[i] = 0.20
                position = 1
            # Short conditions: price breaks below S3 AND price < EMA50(4h) AND volume spike
            elif (close[i] < S3_aligned[i] and 
                  close[i-1] >= S3_aligned[i-1] and  # ensure breakdown just happened
                  close[i] < ema_50_aligned[i] and 
                  volume_filter[i]):
                signals[i] = -0.20
                position = -1
        elif position == 1:
            # Exit long: price retests PP OR price < EMA50(4h) (trend flip)
            if (close[i] <= PP_aligned[i] or 
                close[i] < ema_50_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.20
        elif position == -1:
            # Exit short: price retests PP OR price > EMA50(4h) (trend flip)
            if (close[i] >= PP_aligned[i] or 
                close[i] > ema_50_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.20
    
    return signals
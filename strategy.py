#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1h Camarilla R3/S3 breakout with 4h EMA50 trend filter and volume confirmation
# Camarilla pivot levels (R3/S3) act as strong intraday support/resistance; breakouts capture momentum
# 4h EMA50 ensures alignment with higher timeframe trend to reduce whipsaws
# Volume spike (>1.8x 20-period EMA) confirms breakout authenticity
# Session filter (08-20 UTC) avoids low-liquidity periods
# Target: 15-35 trades/year (60-140 total over 4 years) to minimize fee drag
# Works in bull/bear: trend filter prevents counter-trend trades, breakouts work in ranging markets via reversion to mean after extreme moves

name = "1h_Camarilla_R3S3_Breakout_4hEMA50_VolumeSpike"
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
    open_time = prices['open_time'].values
    
    # Session filter: 08-20 UTC (avoid low-liquidity periods)
    hours = pd.DatetimeIndex(open_time).hour
    in_session = (hours >= 8) & (hours <= 20)
    
    # Get 4h data for EMA trend filter
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 2:
        return np.zeros(n)
    
    # Calculate 4h EMA(50) for trend filter
    close_4h = df_4h['close'].values
    ema_50_4h = pd.Series(close_4h).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_4h_aligned = align_htf_to_ltf(prices, df_4h, ema_50_4h)
    
    # Calculate Camarilla pivot points on 1h data (using previous bar's OHLC)
    # Camarilla: R4 = close + 1.5*(high-low), R3 = close + 1.1*(high-low), 
    #            S3 = close - 1.1*(high-low), S4 = close - 1.5*(high-low)
    # We use R3/S3 as entry levels (more reliable than R4/S4)
    shift_high = np.roll(high, 1)
    shift_low = np.roll(low, 1)
    shift_close = np.roll(close, 1)
    # First bar: use current values (no look-ahead)
    shift_high[0] = high[0]
    shift_low[0] = low[0]
    shift_close[0] = close[0]
    
    camarilla_range = shift_high - shift_low
    camarilla_R3 = shift_close + 1.1 * camarilla_range
    camarilla_S3 = shift_close - 1.1 * camarilla_range
    
    # Volume confirmation: 20-period EMA on 1h volume
    vol_series = pd.Series(volume)
    vol_ema_20 = vol_series.ewm(span=20, adjust=False, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(1, n):  # Start from 1 to have valid previous bar
        # Skip if any value is NaN or outside session
        if (np.isnan(ema_50_4h_aligned[i]) or np.isnan(vol_ema_20[i]) or 
            not in_session[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Volume spike: current volume > 1.8 x 20-period EMA (tight to avoid overtrading)
        volume_spike = volume[i] > (1.8 * vol_ema_20[i])
        
        # Camarilla R3/S3 breakout signals with 4h trend filter
        # Long: price breaks above R3 + price above 4h EMA50 + volume spike
        # Short: price breaks below S3 + price below 4h EMA50 + volume spike
        if position == 0:
            if (close[i] > camarilla_R3[i] and close[i] > ema_50_4h_aligned[i] and volume_spike):
                signals[i] = 0.20
                position = 1
            elif (close[i] < camarilla_S3[i] and close[i] < ema_50_4h_aligned[i] and volume_spike):
                signals[i] = -0.20
                position = -1
        elif position == 1:
            # Exit long: price crosses below S3 (mean reversion) OR below 4h EMA50
            if close[i] < camarilla_S3[i] or close[i] < ema_50_4h_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.20
        elif position == -1:
            # Exit short: price crosses above R3 (mean reversion) OR above 4h EMA50
            if close[i] > camarilla_R3[i] or close[i] > ema_50_4h_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.20
    
    return signals
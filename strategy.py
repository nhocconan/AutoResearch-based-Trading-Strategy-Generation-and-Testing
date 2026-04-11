#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "1h_4h_1d_camarilla_pivot_v1"
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
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    # Load 4h and 1d data ONCE before loop
    df_4h = get_htf_data(prices, '4h')
    df_1d = get_htf_data(prices, '1d')
    
    if len(df_4h) < 10 or len(df_1d) < 10:
        return signals
    
    # Calculate 4h Camarilla pivot levels
    high_4h = df_4h['high'].values
    low_4h = df_4h['low'].values
    close_4h = df_4h['close'].values
    
    pivot_4h = (high_4h + low_4h + close_4h) / 3
    range_4h = high_4h - low_4h
    r1_4h = close_4h + range_4h * 1.1 / 12
    r2_4h = close_4h + range_4h * 1.1 / 6
    r3_4h = close_4h + range_4h * 1.1 / 4
    r4_4h = close_4h + range_4h * 1.1 / 2
    s1_4h = close_4h - range_4h * 1.1 / 12
    s2_4h = close_4h - range_4h * 1.1 / 6
    s3_4h = close_4h - range_4h * 1.1 / 4
    s4_4h = close_4h - range_4h * 1.1 / 2
    
    # Shift by 1 to use only completed 4h bars
    pivot_4h = np.roll(pivot_4h, 1)
    r1_4h = np.roll(r1_4h, 1)
    r2_4h = np.roll(r2_4h, 1)
    r3_4h = np.roll(r3_4h, 1)
    r4_4h = np.roll(r4_4h, 1)
    s1_4h = np.roll(s1_4h, 1)
    s2_4h = np.roll(s2_4h, 1)
    s3_4h = np.roll(s3_4h, 1)
    s4_4h = np.roll(s4_4h, 1)
    pivot_4h[0] = np.nan
    r1_4h[0] = np.nan
    r2_4h[0] = np.nan
    r3_4h[0] = np.nan
    r4_4h[0] = np.nan
    s1_4h[0] = np.nan
    s2_4h[0] = np.nan
    s3_4h[0] = np.nan
    s4_4h[0] = np.nan
    
    # Align 4h indicators to 1h timeframe
    pivot_4h_aligned = align_htf_to_ltf(prices, df_4h, pivot_4h)
    r1_4h_aligned = align_htf_to_ltf(prices, df_4h, r1_4h)
    r2_4h_aligned = align_htf_to_ltf(prices, df_4h, r2_4h)
    r3_4h_aligned = align_htf_to_ltf(prices, df_4h, r3_4h)
    r4_4h_aligned = align_htf_to_ltf(prices, df_4h, r4_4h)
    s1_4h_aligned = align_htf_to_ltf(prices, df_4h, s1_4h)
    s2_4h_aligned = align_htf_to_ltf(prices, df_4h, s2_4h)
    s3_4h_aligned = align_htf_to_ltf(prices, df_4h, s3_4h)
    s4_4h_aligned = align_htf_to_ltf(prices, df_4h, s4_4h)
    
    # Calculate 1d EMA for trend filter
    close_1d = df_1d['close'].values
    ema_50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # Calculate 1h ATR for volatility filter
    tr1 = high[1:] - low[1:]
    tr2 = np.abs(high[1:] - close[:-1])
    tr3 = np.abs(low[1:] - close[:-1])
    tr = np.concatenate([[np.nan], np.maximum(tr1, np.maximum(tr2, tr3))])
    atr = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    # Calculate 1h volume moving average
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    # Pre-compute session hours (08-20 UTC)
    hours = pd.DatetimeIndex(prices['open_time']).hour
    
    for i in range(50, n):
        # Skip if any required data is invalid
        if (np.isnan(pivot_4h_aligned[i]) or np.isnan(r1_4h_aligned[i]) or np.isnan(s1_4h_aligned[i]) or
            np.isnan(ema_50_1d_aligned[i]) or np.isnan(atr[i]) or np.isnan(vol_ma_20[i])):
            signals[i] = 0.0
            continue
        
        # Session filter: 08-20 UTC
        hour = hours[i]
        in_session = (8 <= hour <= 20)
        
        price_close = close[i]
        price_high = high[i]
        price_low = low[i]
        volume_current = volume[i]
        vol_ma = vol_ma_20[i]
        atr_val = atr[i]
        
        # Volume confirmation
        volume_confirmed = volume_current > 1.5 * vol_ma
        
        # Volatility filter: avoid extremely low volatility
        vol_filter = atr_val > 0.005 * price_close  # ATR > 0.5% of price
        
        # Long conditions: price above S1 and S2, with volume and vol filter, in session, above daily EMA
        long_signal = (volume_confirmed and vol_filter and in_session and
                      price_close > s1_4h_aligned[i] and price_close > s2_4h_aligned[i] and
                      price_close > ema_50_1d_aligned[i])
        
        # Short conditions: price below R1 and R2, with volume and vol filter, in session, below daily EMA
        short_signal = (volume_confirmed and vol_filter and in_session and
                       price_close < r1_4h_aligned[i] and price_close < r2_4h_aligned[i] and
                       price_close < ema_50_1d_aligned[i])
        
        # Exit when price crosses the pivot point
        exit_long = position == 1 and price_close < pivot_4h_aligned[i]
        exit_short = position == -1 and price_close > pivot_4h_aligned[i]
        
        # Trading logic
        if long_signal and position != 1:
            position = 1
            signals[i] = 0.20
        elif short_signal and position != -1:
            position = -1
            signals[i] = -0.20
        elif position == 1 and exit_long:
            position = 0
            signals[i] = 0.0
        elif position == -1 and exit_short:
            position = 0
            signals[i] = 0.0
        else:
            # Maintain current position
            signals[i] = 0.20 if position == 1 else (-0.20 if position == -1 else 0.0)
    
    return signals

# Hypothesis: 1h Camarilla pivot strategy with 4h/1d confirmation
# Uses 4h Camarilla levels (S1, S2, R1, R2) for entry and pivot for exit
# Filters: 1d EMA50 for trend, volume > 1.5x average, ATR > 0.5% price, session 08-20 UTC
# Long when price > S1 and S2 with filters; short when price < R1 and R2 with filters
# Exit when price crosses 4h pivot point
# Designed for 60-150 total trades over 4 years (15-37/year) to minimize fee drift
# Works in both bull and bear markets by combining mean-reversion at pivot levels with trend filter
# Camarilla levels provide high-probability reversal points; 4h/1d context avoids counter-trend trades
# Session filter reduces noise during low-liquidity hours
# Position size fixed at 0.20 to control drawdown and enable consistent scaling
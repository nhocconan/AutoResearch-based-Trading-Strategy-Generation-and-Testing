#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "1h_4h_1d_supertrend_follow_v1"
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
    
    # Load 4h and 1d data ONCE before loop
    df_4h = get_htf_data(prices, '4h')
    df_1d = get_htf_data(prices, '1d')
    
    if len(df_4h) < 30 or len(df_1d) < 30:
        return np.zeros(n)
    
    # Calculate 4h Supertrend (ATR=10, multiplier=3)
    high_4h = df_4h['high'].values
    low_4h = df_4h['low'].values
    close_4h = df_4h['close'].values
    
    # True Range for 4h
    tr1_4h = high_4h[1:] - low_4h[1:]
    tr2_4h = np.abs(high_4h[1:] - close_4h[:-1])
    tr3_4h = np.abs(low_4h[1:] - close_4h[:-1])
    tr_4h = np.concatenate([[np.nan], np.maximum(tr1_4h, np.maximum(tr2_4h, tr3_4h))])
    atr_4h = pd.Series(tr_4h).rolling(window=10, min_periods=10).mean().values
    
    # Basic Upper and Lower Bands
    hl2_4h = (high_4h + low_4h) / 2
    upper_band_4h = hl2_4h + 3.0 * atr_4h
    lower_band_4h = hl2_4h - 3.0 * atr_4h
    
    # Initialize Supertrend
    supertrend_4h = np.full_like(close_4h, np.nan)
    direction_4h = np.ones_like(close_4h)  # 1 for uptrend, -1 for downtrend
    
    for i in range(1, len(close_4h)):
        if np.isnan(atr_4h[i-1]) or np.isnan(upper_band_4h[i-1]) or np.isnan(lower_band_4h[i-1]):
            supertrend_4h[i] = np.nan
            direction_4h[i] = direction_4h[i-1] if i > 0 else 1
            continue
            
        if close_4h[i] > upper_band_4h[i-1]:
            direction_4h[i] = 1
        elif close_4h[i] < lower_band_4h[i-1]:
            direction_4h[i] = -1
        else:
            direction_4h[i] = direction_4h[i-1]
            if direction_4h[i] == 1 and lower_band_4h[i] < lower_band_4h[i-1]:
                lower_band_4h[i] = lower_band_4h[i-1]
            if direction_4h[i] == -1 and upper_band_4h[i] > upper_band_4h[i-1]:
                upper_band_4h[i] = upper_band_4h[i-1]
        
        supertrend_4h[i] = lower_band_4h[i] if direction_4h[i] == 1 else upper_band_4h[i]
    
    # Align 4h Supertrend direction to 1h
    supertrend_dir_1h = align_htf_to_ltf(prices, df_4h, direction_4h)
    
    # 1d EMA50 for trend filter
    ema_50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # 1h ATR for volatility filter
    tr1 = high[1:] - low[1:]
    tr2 = np.abs(high[1:] - close[:-1])
    tr3 = np.abs(low[1:] - close[:-1])
    tr = np.concatenate([[np.nan], np.maximum(tr1, np.maximum(tr2, tr3))])
    atr_1h = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    # 1h volume filter: volume > 1.2x 30-period average
    vol_ma_30 = pd.Series(volume).rolling(window=30, min_periods=30).mean().values
    
    # Session filter: 8-20 UTC
    hours = pd.DatetimeIndex(prices['open_time']).hour
    in_session = (hours >= 8) & (hours <= 20)
    
    # Minimum holding period: 4 bars (4 hours) to reduce churn
    hold_count = np.zeros(n, dtype=int)
    
    signals = np.zeros(n)
    position = 0
    
    for i in range(50, n):
        if hold_count[i] > 0:
            hold_count[i] -= 1
        
        if (np.isnan(supertrend_dir_1h[i]) or np.isnan(ema_50_1d_aligned[i]) or
            np.isnan(atr_1h[i]) or np.isnan(vol_ma_30[i]) or
            not in_session[i] or hold_count[i] > 0):
            signals[i] = 0.0 if position == 0 else (0.20 if position == 1 else -0.20)
            continue
        
        price_close = close[i]
        price_high = high[i]
        price_low = low[i]
        volume_current = volume[i]
        vol_ma = vol_ma_30[i]
        
        volume_confirmed = volume_current > 1.2 * vol_ma
        
        # Long: 4h uptrend, price above 1d EMA50, volume confirmation
        long_signal = (supertrend_dir_1h[i] == 1) and (price_close > ema_50_1d_aligned[i]) and volume_confirmed
        
        # Short: 4h downtrend, price below 1d EMA50, volume confirmation
        short_signal = (supertrend_dir_1h[i] == -1) and (price_close < ema_50_1d_aligned[i]) and volume_confirmed
        
        # Exit when 4h trend changes
        exit_long = position == 1 and supertrend_dir_1h[i] == -1
        exit_short = position == -1 and supertrend_dir_1h[i] == 1
        
        if long_signal and position != 1:
            position = 1
            hold_count[i] = 4
            signals[i] = 0.20
        elif short_signal and position != -1:
            position = -1
            hold_count[i] = 4
            signals[i] = -0.20
        elif position == 1 and exit_long:
            position = 0
            signals[i] = 0.0
        elif position == -1 and exit_short:
            position = 0
            signals[i] = 0.0
        else:
            signals[i] = 0.20 if position == 1 else (-0.20 if position == -1 else 0.0)
    
    return signals

# Hypothesis: 1h Supertrend following with 4h trend filter and 1d EMA50.
# Uses 4h Supertrend (ATR=10, multiplier=3) as primary trend filter - only trade in direction of 4h trend.
# 1d EMA50 ensures alignment with daily trend to avoid counter-trend trades in strong moves.
# Volume confirmation (>1.2x 30-period average) filters weak breakouts.
# Session filter (8-20 UTC) reduces noise during low-liquidity hours.
# Minimum holding period of 4 bars prevents churn and allows trends to develop.
# Position size: 0.20 for risk management.
# Target: 60-120 total trades over 4 years (15-30/year) to minimize fee drag.
# Works in bull markets (follow 4h uptrends) and bear markets (follow 4h downtrends).
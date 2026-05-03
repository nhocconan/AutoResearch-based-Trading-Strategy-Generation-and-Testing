#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1h Camarilla H3/L3 breakout with 4h EMA50 trend filter and volume spike confirmation.
# Long when price breaks above 1h Camarilla H3 level in 4h uptrend (price > EMA50) with volume spike.
# Short when price breaks below 1h Camarilla L3 level in 4h downtrend (price < EMA50) with volume spike.
# Uses discrete sizing 0.20 to balance return and drawdown. Target: 60-150 total trades over 4 years.
# Camarilla H3/L3 levels provide strong support/resistance, 4h EMA50 ensures higher timeframe alignment,
# Volume spike confirms institutional interest. Session filter (08-20 UTC) reduces noise trades.
# Works in both bull and bear markets by only trading with the 4h trend, avoiding counter-trend whipsaws.

name = "1h_Camarilla_H3L3_4hEMA50_VolumeSpike_Session"
timeframe = "1h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    open_time = prices['open_time'].values
    
    # Pre-compute session filter (08-20 UTC)
    hours = pd.DatetimeIndex(open_time).hour
    in_session = (hours >= 8) & (hours <= 20)
    
    # Get 1h data for Camarilla calculation
    df_1h = get_htf_data(prices, '1h')
    
    if len(df_1h) < 20:
        return np.zeros(n)
    
    # Calculate 1h Camarilla levels (based on previous bar's high/low/close)
    high_1h = df_1h['high'].values
    low_1h = df_1h['low'].values
    close_1h = df_1h['close'].values
    
    # Camarilla H3 and L3 levels
    # H3 = close + (high - low) * 1.1/2
    # L3 = close - (high - low) * 1.1/2
    camarilla_h3_1h = close_1h + (high_1h - low_1h) * 1.1 / 2
    camarilla_l3_1h = close_1h - (high_1h - low_1h) * 1.1 / 2
    
    # Align Camarilla levels to 1h timeframe (already 1h, but using helper for consistency)
    camarilla_h3_aligned = align_htf_to_ltf(prices, df_1h, camarilla_h3_1h)
    camarilla_l3_aligned = align_htf_to_ltf(prices, df_1h, camarilla_l3_1h)
    
    # Get 4h data for trend filter
    df_4h = get_htf_data(prices, '4h')
    
    if len(df_4h) < 50:
        return np.zeros(n)
    
    # Calculate 4h EMA50
    close_4h = df_4h['close'].values
    ema_50_4h = pd.Series(close_4h).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_4h_aligned = align_htf_to_ltf(prices, df_4h, ema_50_4h)
    
    # Volume spike detection (20-period volume MA)
    volume_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (volume_ma * 2.0)  # Volume at least 2x average
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(100, n):
        # Skip if any value is NaN
        if (np.isnan(camarilla_h3_aligned[i]) or np.isnan(camarilla_l3_aligned[i]) or 
            np.isnan(ema_50_4h_aligned[i]) or np.isnan(volume_ma[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
            
        # Session filter
        if not in_session[i]:
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
            
        close_val = close[i]
        vol_spike = volume_spike[i]
        trend_up = close_val > ema_50_4h_aligned[i]   # 4h uptrend
        trend_down = close_val < ema_50_4h_aligned[i]  # 4h downtrend
        
        if position == 0:
            # Long: price breaks above Camarilla H3 AND 4h uptrend AND volume spike
            if close_val > camarilla_h3_aligned[i] and trend_up and vol_spike:
                signals[i] = 0.20
                position = 1
            # Short: price breaks below Camarilla L3 AND 4h downtrend AND volume spike
            elif close_val < camarilla_l3_aligned[i] and trend_down and vol_spike:
                signals[i] = -0.20
                position = -1
        elif position == 1:
            # Exit long: price breaks below Camarilla L3 (reversal signal)
            if close_val < camarilla_l3_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.20
        elif position == -1:
            # Exit short: price breaks above Camarilla H3 (reversal signal)
            if close_val > camarilla_h3_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.20
    
    return signals
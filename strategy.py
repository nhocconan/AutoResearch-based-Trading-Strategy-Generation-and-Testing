#!/usr/bin/env python3
# Hypothesis: 1h Camarilla R1/S1 breakout with 4h EMA50 trend filter and 1d volume spike confirmation (>2.0x 20-period average). Uses discrete position sizing (0.0, ±0.20) to minimize fee churn. Session filter (08-20 UTC) reduces noise. Designed for 1h timeframe with 4h/1d HTF filters to achieve 15-35 trades/year.

name = "1h_Camarilla_R1S1_Breakout_4hEMA50_1dVolume_v1"
timeframe = "1h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    open_ = prices['open'].values
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Pre-compute session hours (08-20 UTC)
    hours = pd.DatetimeIndex(prices["open_time"]).hour
    in_session = (hours >= 8) & (hours <= 20)
    
    # --- 1h Indicators (LTF) ---
    # Volume confirmation: > 1.5x 20-period average
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_confirm = volume > (1.5 * vol_ma_20)
    
    # --- 4h Indicators (HTF) ---
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 50:
        return np.zeros(n)
    close_4h = df_4h['close'].values
    
    # 4h EMA(50) - trend filter
    ema_50_4h = pd.Series(close_4h).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_4h_aligned = align_htf_to_ltf(prices, df_4h, ema_50_4h)
    
    # --- 1d Indicators (HTF) ---
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    volume_1d = df_1d['volume'].values
    
    # 1d volume spike: > 2.0x 20-period average
    vol_ma_20_1d = pd.Series(volume_1d).rolling(window=20, min_periods=20).mean().values
    volume_spike_1d = volume_1d > (2.0 * vol_ma_20_1d)
    volume_spike_1d_aligned = align_htf_to_ltf(prices, df_1d, volume_spike_1d)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(1, n):
        # Skip if missing data or outside session
        if (np.isnan(ema_50_4h_aligned[i]) or
            np.isnan(volume_spike_1d_aligned[i]) or
            np.isnan(volume_confirm[i]) or
            not in_session[i]):
            signals[i] = 0.0
            continue
        
        # Calculate Camarilla levels for today (using previous bar's OHLC)
        if i >= 1:
            prev_high = high[i-1]
            prev_low = low[i-1]
            prev_close = close[i-1]
            range_ = prev_high - prev_low
            
            # Camarilla levels (R1/S1 = inner levels for breakout)
            R1 = prev_close + range_ * 1.1 / 12
            S1 = prev_close - range_ * 1.1 / 12
        else:
            R1 = np.nan
            S1 = np.nan
        
        if position == 0:
            # LONG: Price breaks above R1 AND close > 4h EMA50 (bullish trend) AND 1d volume spike AND volume confirm
            if (not np.isnan(R1) and 
                close[i] > R1 and 
                close[i] > ema_50_4h_aligned[i] and 
                volume_spike_1d_aligned[i] and
                volume_confirm[i]):
                signals[i] = 0.20
                position = 1
            # SHORT: Price breaks below S1 AND close < 4h EMA50 (bearish trend) AND 1d volume spike AND volume confirm
            elif (not np.isnan(S1) and 
                  close[i] < S1 and 
                  close[i] < ema_50_4h_aligned[i] and 
                  volume_spike_1d_aligned[i] and
                  volume_confirm[i]):
                signals[i] = -0.20
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: Price crosses below 4h EMA50 (trend change) OR touches S1 (mean reversion)
            if close[i] < ema_50_4h_aligned[i] or close[i] < S1:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.20
        elif position == -1:
            # EXIT SHORT: Price crosses above 4h EMA50 (trend change) OR touches R1 (mean reversion)
            if close[i] > ema_50_4h_aligned[i] or close[i] > R1:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.20
    
    return signals
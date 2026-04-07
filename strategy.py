#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1-hour RSI(14) with 4-hour trend filter (EMA25/EMA50) and 1-day volume confirmation
# Targets 60-150 total trades over 4 years (15-37/year) to minimize fee drag
# Uses session filter (08-20 UTC) to avoid low-liquidity periods
# Works in bull/bear via trend filter and mean-reversion RSI signals
# Entry: RSI < 30 in uptrend or RSI > 70 in downtrend with volume confirmation
# Exit: RSI crosses 50 (mean reversion completion) or opposite extreme

name = "1h_rsi_trend_filter_vol_session_v1"
timeframe = "1h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Price data
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # 4-hour data for trend filter
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 50:
        return np.zeros(n)
    
    # Calculate 4h EMA(25) and EMA(50) for trend filter
    close_4h = df_4h['close'].values
    ema_25_4h = pd.Series(close_4h).ewm(span=25, adjust=False, min_periods=25).mean().values
    ema_50_4h = pd.Series(close_4h).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_25_4h_aligned = align_htf_to_ltf(prices, df_4h, ema_25_4h)
    ema_50_4h_aligned = align_htf_to_ltf(prices, df_4h, ema_50_4h)
    
    # 1-day data for volume confirmation
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    
    # Calculate 1d average volume (20-period)
    vol_1d = df_1d['volume'].values
    vol_avg_1d = pd.Series(vol_1d).rolling(window=20, min_periods=20).mean().values
    vol_avg_1d_aligned = align_htf_to_ltf(prices, df_1d, vol_avg_1d)
    
    # RSI(14) calculation
    delta = np.diff(close, prepend=close[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    avg_gain = pd.Series(gain).ewm(span=14, adjust=False, min_periods=14).mean().values
    avg_loss = pd.Series(loss).ewm(span=14, adjust=False, min_periods=14).mean().values
    rs = avg_gain / (avg_loss + 1e-10)
    rsi = 100 - (100 / (1 + rs))
    
    # Session filter: 08-20 UTC (pre-compute hours)
    hours = pd.DatetimeIndex(prices['open_time']).hour
    in_session = (hours >= 8) & (hours <= 20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(50, n):
        # Skip if required data not available
        if (np.isnan(ema_25_4h_aligned[i]) or np.isnan(ema_50_4h_aligned[i]) or 
            np.isnan(rsi[i]) or np.isnan(vol_avg_1d_aligned[i])):
            if position != 0:
                signals[i] = position * 0.20
            else:
                signals[i] = 0.0
            continue
        
        # Session filter
        if not in_session[i]:
            if position != 0:
                signals[i] = position * 0.20
            else:
                signals[i] = 0.0
            continue
        
        if position == 1:  # long position
            # Exit: RSI crosses above 50 (mean reversion complete)
            if rsi[i] > 50 and rsi[i-1] <= 50:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.20
        elif position == -1:  # short position
            # Exit: RSI crosses below 50 (mean reversion complete)
            if rsi[i] < 50 and rsi[i-1] >= 50:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.20
        else:
            # Trend filter: 4h EMA(25) > EMA(50) for uptrend, < for downtrend
            uptrend = ema_25_4h_aligned[i] > ema_50_4h_aligned[i]
            downtrend = ema_25_4h_aligned[i] < ema_50_4h_aligned[i]
            
            # Volume confirmation: current volume > 1.5 * 1-day average volume
            volume_confirm = volume[i] > 1.5 * vol_avg_1d_aligned[i]
            
            # Long: RSI < 30 (oversold) in uptrend with volume
            if rsi[i] < 30 and uptrend and volume_confirm:
                signals[i] = 0.20
                position = 1
            # Short: RSI > 70 (overbought) in downtrend with volume
            elif rsi[i] > 70 and downtrend and volume_confirm:
                signals[i] = -0.20
                position = -1
    
    return signals
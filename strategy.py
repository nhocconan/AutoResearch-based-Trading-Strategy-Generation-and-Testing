#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12-hour price action combined with daily RSI and volume confirmation
# Long when price crosses above daily VWAP with RSI < 40 (oversold) and volume spike
# Short when price crosses below daily VWAP with RSI > 60 (overbought) and volume spike
# Daily VWAP acts as dynamic support/resistance; RSI filters for momentum exhaustion
# Volume spike confirms institutional participation; avoids false signals
# Targets 50-150 total trades over 4 years (12-37/year) to balance opportunity and cost

name = "12h_DailyVWAP_RSI_Volume"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Get daily data once for VWAP and RSI
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    # Calculate daily VWAP (typical price * volume / cumulative volume)
    typical_price = (df_1d['high'] + df_1d['low'] + df_1d['close']) / 3.0
    vwap_num = (typical_price * df_1d['volume']).cumsum()
    vwap_den = df_1d['volume'].cumsum()
    daily_vwap = (vwap_num / vwap_den).replace([np.inf, -np.inf], np.nan).ffill().bfill().values
    daily_vwap_aligned = align_htf_to_ltf(prices, df_1d, daily_vwap)
    
    # Calculate daily RSI(14)
    daily_close = df_1d['close'].values
    delta = np.diff(daily_close, prepend=daily_close[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    avg_gain = pd.Series(gain).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    avg_loss = pd.Series(loss).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    rs = np.where(avg_loss != 0, avg_gain / avg_loss, 0)
    daily_rsi = 100 - (100 / (1 + rs))
    daily_rsi_aligned = align_htf_to_ltf(prices, df_1d, daily_rsi)
    
    # Volume spike: current volume > 2.0 * 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (2.0 * vol_ma)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 100  # warmup for calculations
    
    for i in range(start_idx, n):
        # Skip if any critical data is NaN
        if (np.isnan(daily_vwap_aligned[i]) or np.isnan(daily_rsi_aligned[i]) or 
            np.isnan(vol_ma[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        vwap = daily_vwap_aligned[i]
        rsi = daily_rsi_aligned[i]
        vol_spike = volume_spike[i]
        
        if position == 0:
            # Enter long: price crosses above daily VWAP, RSI < 40 (oversold), volume spike
            if close[i] > vwap and close[i-1] <= vwap and rsi < 40 and vol_spike:
                signals[i] = 0.25
                position = 1
            # Enter short: price crosses below daily VWAP, RSI > 60 (overbought), volume spike
            elif close[i] < vwap and close[i-1] >= vwap and rsi > 60 and vol_spike:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: price crosses below daily VWAP or RSI > 70
            if close[i] < vwap or rsi > 70:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: price crosses above daily VWAP or RSI < 30
            if close[i] > vwap or rsi < 30:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals
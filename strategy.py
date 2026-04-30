#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h KAMA trend + 1d RSI mean reversion + volume spike confirmation
# KAMA adapts to market noise - trends efficiently in trending markets, stays flat in chop
# 1d RSI < 30 for long, > 70 for short provides mean-reversion edge in bear markets
# Volume spike (2.0x 20-period average) confirms momentum behind moves
# Discrete sizing 0.25 minimizes fee churn. Target: 12-37 trades/year (50-150 total over 4 years)
# Works in bull via KAMA longs with uptrend confirmation, in bear via RSI extremes with volume

name = "12h_KAMA_1dRSI_VolumeSpike_v1"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    open_time = prices['open_time'].values
    
    # Pre-compute session hours (08-20 UTC) to avoid datetime errors
    hours = pd.DatetimeIndex(open_time).hour
    in_session = (hours >= 8) & (hours <= 20)
    
    # Calculate 1d RSI for mean reversion
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 14:
        return np.zeros(n)
    delta = pd.Series(df_1d['close'].values).diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(alpha=1/14, adjust=False, min_periods=14).mean()
    avg_loss = loss.ewm(alpha=1/14, adjust=False, min_periods=14).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    rsi_1d = 100 - (100 / (1 + rs))
    rsi_1d = rsi_1d.fillna(50).values  # neutral when undefined
    rsi_1d_aligned = align_htf_to_ltf(prices, df_1d, rsi_1d)
    
    # Calculate KAMA (12h timeframe) - adaptive trend indicator
    # Efficiency Ratio: ER = |net change| / sum of absolute changes
    change = np.abs(np.diff(close, n=10))
    volatility = np.sum(np.abs(np.diff(close)), axis=0)  # placeholder - will compute properly below
    # Recalculate properly:
    er = np.zeros(n)
    for i in range(10, n):
        net_change = abs(close[i] - close[i-10])
        total_change = np.sum(np.abs(np.diff(close[i-10:i+1])))
        er[i] = net_change / total_change if total_change > 0 else 0
    # Smoothing constants
    fastest = 2 / (2 + 1)   # EMA(2)
    slowest = 2 / (30 + 1)  # EMA(30)
    sc = (er * (fastest - slowest) + slowest) ** 2
    # Initialize KAMA
    kama = np.zeros(n)
    kama[:10] = close[:10]
    for i in range(10, n):
        kama[i] = kama[i-1] + sc[i] * (close[i] - kama[i-1])
    
    # Volume confirmation: volume > 2.0x 20-period average
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (2.0 * vol_ma_20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    start_idx = max(20, 10, 14)  # warmup for volume MA, KAMA, and 1d RSI
    
    for i in range(start_idx, n):
        # Skip if indicators not ready
        if (np.isnan(vol_ma_20[i]) or np.isnan(rsi_1d_aligned[i])):
            signals[i] = 0.0
            continue
            
        # Session filter: only trade 08-20 UTC
        if not in_session[i]:
            signals[i] = 0.0
            continue
            
        curr_close = close[i]
        curr_kama = kama[i]
        curr_rsi = rsi_1d_aligned[i]
        curr_volume_spike = volume_spike[i]
        
        if position == 0:  # Flat - look for new entries
            # Require volume spike
            if curr_volume_spike:
                # Long entry: price > KAMA (uptrend) AND 1d RSI < 30 (oversold mean reversion)
                if curr_close > curr_kama and curr_rsi < 30:
                    signals[i] = 0.25
                    position = 1
                    entry_price = curr_close
                # Short entry: price < KAMA (downtrend) AND 1d RSI > 70 (overbought mean reversion)
                elif curr_close < curr_kama and curr_rsi > 70:
                    signals[i] = -0.25
                    position = -1
                    entry_price = curr_close
        
        elif position == 1:  # Long position
            # Exit when price crosses below KAMA (trend change) OR RSI reaches extremes
            if curr_close < curr_kama or curr_rsi > 70:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:  # Short position
            # Exit when price crosses above KAMA (trend change) OR RSI reaches extremes
            if curr_close > curr_kama or curr_rsi < 30:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals
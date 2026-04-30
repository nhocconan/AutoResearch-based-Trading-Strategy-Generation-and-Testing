#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h KAMA trend direction + 1d RSI(14) mean reversion + volume spike
# KAMA adapts to market noise - tracks trend efficiently in both bull and bear markets
# 1d RSI < 30 = oversold (long bias), RSI > 70 = overbought (short bias) on higher timeframe
# Volume spike (2.0x 20-period average) confirms momentum for entry
# Discrete sizing 0.25 minimizes fee churn. Target: 12-37 trades/year (50-150 total over 4 years).
# Works in bull via KAMA uptrend + RSI mean reversion longs, in bear via KAMA downtrend + RSI mean reversion shorts.

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
    
    # Calculate 1d RSI(14) for mean reversion filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 14:
        return np.zeros(n)
    close_1d = pd.Series(df_1d['close'].values)
    delta = close_1d.diff()
    gain = delta.where(delta > 0, 0.0)
    loss = -delta.where(delta < 0, 0.0)
    avg_gain = gain.ewm(alpha=1/14, adjust=False, min_periods=14).mean()
    avg_loss = loss.ewm(alpha=1/14, adjust=False, min_periods=14).mean()
    rs = avg_gain / avg_loss
    rsi_1d = 100 - (100 / (1 + rs))
    rsi_1d_values = rsi_1d.values
    rsi_1d_aligned = align_htf_to_ltf(prices, df_1d, rsi_1d_values)
    
    # Calculate KAMA(10, 2, 30) for trend direction
    # ER = |net change| / sum(|changes|)
    change = np.abs(np.diff(close, prepend=close[0]))
    volatility = np.sum(np.abs(np.diff(close, prepend=close[0])))

    # Avoid division by zero
    er = np.zeros_like(change)
    er[volatility != 0] = change[volatility != 0] / volatility[volatility != 0]
    
    # Smoothing constants
    sc = (er * (2/(2+1) - 2/(30+1)) + 2/(30+1)) ** 2
    kama = np.zeros_like(close)
    kama[0] = close[0]
    
    for i in range(1, len(close)):
        kama[i] = kama[i-1] + sc[i] * (close[i] - kama[i-1])
    
    # Volume confirmation: volume > 2.0x 20-period average
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (2.0 * vol_ma_20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    start_idx = max(20, 30)  # warmup for volume MA and KAMA
    
    for i in range(start_idx, n):
        # Skip if indicators not ready
        if (np.isnan(kama[i]) or np.isnan(rsi_1d_aligned[i]) or np.isnan(vol_ma_20[i])):
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
                # Bullish entry: price > KAMA (uptrend) AND RSI < 30 (oversold)
                if curr_close > curr_kama and curr_rsi < 30:
                    signals[i] = 0.25
                    position = 1
                    entry_price = curr_close
                # Bearish entry: price < KAMA (downtrend) AND RSI > 70 (overbought)
                elif curr_close < curr_kama and curr_rsi > 70:
                    signals[i] = -0.25
                    position = -1
                    entry_price = curr_close
        
        elif position == 1:  # Long position
            # Exit when price crosses below KAMA OR RSI > 50 (mean reversion complete)
            if curr_close < curr_kama or curr_rsi > 50:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:  # Short position
            # Exit when price crosses above KAMA OR RSI < 50 (mean reversion complete)
            if curr_close > curr_kama or curr_rsi < 50:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals
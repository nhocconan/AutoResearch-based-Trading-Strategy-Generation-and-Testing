#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1h RSI(14) mean reversion with 4h EMA50 trend filter and 1d volume spike confirmation
# In ranging markets (BTC/ETH 2025+), RSI extremes (>70/<30) often reverse toward the mean.
# 4h EMA50 ensures we only take mean-reversion trades in the direction of the intermediate trend.
# 1d volume spike (>2.0x 20-period EMA) confirms institutional participation, reducing false signals.
# Session filter (08-20 UTC) reduces noise. Target: 15-35 trades/year with discrete sizing (0.20).

name = "1h_RSI14_4hEMA50_1dVolumeSpike_MR"
timeframe = "1h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    open_time = prices['open_time']
    
    # Session filter: 08-20 UTC (pre-compute to avoid datetime64 issues)
    hours = pd.DatetimeIndex(open_time).hour
    in_session = (hours >= 8) & (hours <= 20)
    
    # Get 4h data for EMA50 trend filter
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 50:
        return np.zeros(n)
    
    # Calculate 4h EMA(50) for trend filter
    close_4h = df_4h['close'].values
    ema_50_4h = pd.Series(close_4h).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_4h_aligned = align_htf_to_ltf(prices, df_4h, ema_50_4h)
    
    # Get 1d data for volume spike confirmation
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # Calculate 1d volume EMA(20) for spike detection
    volume_1d = df_1d['volume'].values
    vol_ema_20_1d = pd.Series(volume_1d).ewm(span=20, adjust=False, min_periods=20).mean().values
    vol_ema_20_1d_aligned = align_htf_to_ltf(prices, df_1d, vol_ema_20_1d)
    
    # Calculate 1h RSI(14) for mean reversion signals
    delta = np.diff(close, prepend=close[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    
    # Wilder's smoothing for RSI
    def wilders_smoothing(data, period):
        result = np.full_like(data, np.nan)
        if len(data) < period:
            return result
        # First value: simple average
        result[period-1] = np.nanmean(data[1:period])
        # Subsequent values: Wilder's smoothing
        for i in range(period, len(data)):
            if np.isnan(result[i-1]) or np.isnan(data[i]):
                result[i] = np.nan
            else:
                result[i] = (result[i-1] * (period-1) + data[i]) / period
        return result
    
    gain_smooth = wilders_smoothing(gain, 14)
    loss_smooth = wilders_smoothing(loss, 14)
    rs = gain_smooth / loss_smooth
    rs = np.where(loss_smooth == 0, 100, rs)
    rsi = 100 - (100 / (1 + rs))
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(14, n):  # Start from 14 to have valid RSI
        # Skip if any value is NaN or outside session
        if (np.isnan(rsi[i]) or np.isnan(ema_50_4h_aligned[i]) or np.isnan(vol_ema_20_1d_aligned[i]) or 
            not in_session[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Volume spike: current 1h volume > 2.0 x 1d volume EMA(20)
        volume_spike = volume[i] > (2.0 * vol_ema_20_1d_aligned[i])
        
        # Trend filter: price above/below 4h EMA50
        price_above_ema = close[i] > ema_50_4h_aligned[i]
        price_below_ema = close[i] < ema_50_4h_aligned[i]
        
        if position == 0:
            # Long: RSI < 30 (oversold) + price above 4h EMA50 (uptrend) + volume spike
            if rsi[i] < 30 and price_above_ema and volume_spike:
                signals[i] = 0.20
                position = 1
            # Short: RSI > 70 (overbought) + price below 4h EMA50 (downtrend) + volume spike
            elif rsi[i] > 70 and price_below_ema and volume_spike:
                signals[i] = -0.20
                position = -1
        elif position == 1:
            # Exit long: RSI > 50 (mean reversion complete) or loses uptrend
            if rsi[i] > 50 or not price_above_ema:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.20
        elif position == -1:
            # Exit short: RSI < 50 (mean reversion complete) or loses downtrend
            if rsi[i] < 50 or not price_below_ema:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.20
    
    return signals
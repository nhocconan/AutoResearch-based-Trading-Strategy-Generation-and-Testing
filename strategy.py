#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h RSI(14) mean reversion with 1d trend filter and volume confirmation
# In 6b markets, RSI extremes often reverse within 1-3 bars. We use 1d EMA50 to filter
# direction (only long when above EMA50, short when below) to avoid counter-trend trades.
# Volume spike confirms participation. This combination reduces false signals while
# capturing mean reversion in both bull and bear markets. Targets 15-25 trades/year.

name = "6h_RSI14_1dEMA50_Volume"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 60:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # RSI(14) on 6h
    delta = np.diff(close, prepend=close[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    
    def wilder_smooth(arr, period):
        result = np.full_like(arr, np.nan)
        if len(arr) < period:
            return result
        result[period-1] = np.nansum(arr[:period])
        for i in range(period, len(arr)):
            result[i] = result[i-1] - (result[i-1] / period) + arr[i]
        return result
    
    gain_smooth = wilder_smooth(gain, 14)
    loss_smooth = wilder_smooth(loss, 14)
    rs = np.where(loss_smooth != 0, gain_smooth / loss_smooth, 0)
    rsi = 100 - (100 / (1 + rs))
    
    # Get 1d data for EMA50 and volume
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    volume_1d = df_1d['volume'].values
    
    # EMA50 on daily
    ema_50 = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Volume spike (2x 20-period MA)
    vol_ma_20 = pd.Series(volume_1d).rolling(window=20, min_periods=20).mean().values
    vol_spike = volume_1d > (vol_ma_20 * 2.0)
    
    # Align to 6h
    ema_50_aligned = align_htf_to_ltf(prices, df_1d, ema_50)
    vol_spike_aligned = align_htf_to_ltf(prices, df_1d, vol_spike)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(14, 50)  # RSI and EMA warmup
    
    for i in range(start_idx, n):
        # Skip if any critical data is NaN
        if (np.isnan(rsi[i]) or np.isnan(ema_50_aligned[i]) or 
            np.isnan(vol_spike_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Enter long: RSI < 30 (oversold), price above 1d EMA50, volume spike
            if rsi[i] < 30 and close[i] > ema_50_aligned[i] and vol_spike_aligned[i]:
                signals[i] = 0.25
                position = 1
            # Enter short: RSI > 70 (overbought), price below 1d EMA50, volume spike
            elif rsi[i] > 70 and close[i] < ema_50_aligned[i] and vol_spike_aligned[i]:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: RSI > 50 (mean reversion complete) or price crosses below EMA50
            if rsi[i] > 50 or close[i] < ema_50_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: RSI < 50 or price crosses above EMA50
            if rsi[i] < 50 or close[i] > ema_50_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals
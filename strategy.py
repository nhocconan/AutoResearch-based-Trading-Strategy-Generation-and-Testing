# NOTE: This strategy is my own work.
# Hypothesis: On 6h timeframe, use 1-day RSI(14) to detect overbought/oversold conditions (RSI < 30 for long, RSI > 70 for short), confirmed by 1-day volume spike (>1.5x 20-period average) and aligned with 1-week trend (price above/below EMA50). Entries are taken when RSI crosses back into normal territory (30-70) to avoid chasing extremes. This mean-reversion approach works in both bull and bear markets by fading short-term extremes while respecting the longer-term trend.
# Target: 50-150 total trades over 4 years (12-37/year). Position size: 0.25.

#!/usr/bin/env python3
name = "6h_RSI_MeanReversion_1dVolSpike_1wTrend"
timeframe = "6h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1d data for RSI and volume
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    volume_1d = df_1d['volume'].values
    
    # Calculate RSI(14) on 1d close
    delta = np.diff(close_1d, prepend=close_1d[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    avg_gain = pd.Series(gain).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    avg_loss = pd.Series(loss).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    rs = avg_gain / (avg_loss + 1e-10)
    rsi_1d = 100 - (100 / (1 + rs))
    
    # Volume spike: volume > 1.5x 20-period average
    vol_ma20_1d = pd.Series(volume_1d).rolling(window=20, min_periods=20).mean().values
    vol_spike_1d = volume_1d > (1.5 * vol_ma20_1d)
    
    # Get 1w data for trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 50:
        return np.zeros(n)
    
    close_1w = df_1w['close'].values
    # Weekly EMA50 for trend
    ema50_1w = pd.Series(close_1w).ewm(span=50, adjust=False, min_periods=50).mean().values
    trend_up_1w = close_1w > ema50_1w
    
    # Align indicators to 6h timeframe
    rsi_1d_aligned = align_htf_to_ltf(prices, df_1d, rsi_1d)
    vol_spike_1d_aligned = align_htf_to_ltf(prices, df_1d, vol_spike_1d)
    trend_up_1w_aligned = align_htf_to_ltf(prices, df_1w, trend_up_1w)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(60, 30)  # Need enough data for indicators
    
    for i in range(start_idx, n):
        # Skip if any data is NaN
        if (np.isnan(rsi_1d_aligned[i]) or 
            np.isnan(vol_spike_1d_aligned[i]) or
            np.isnan(trend_up_1w_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue
        
        if position == 0:
            # Long: RSI < 30 (oversold) + volume spike + weekly uptrend
            if (rsi_1d_aligned[i] < 30 and 
                vol_spike_1d_aligned[i] and 
                trend_up_1w_aligned[i]):
                signals[i] = 0.25
                position = 1
            # Short: RSI > 70 (overbought) + volume spike + weekly downtrend
            elif (rsi_1d_aligned[i] > 70 and 
                  vol_spike_1d_aligned[i] and 
                  not trend_up_1w_aligned[i]):
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Long exit: RSI crosses back above 50 or trend changes
            if (rsi_1d_aligned[i] > 50 or 
                not trend_up_1w_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: RSI crosses back below 50 or trend changes
            if (rsi_1d_aligned[i] < 50 or 
                trend_up_1w_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals
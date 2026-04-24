#!/usr/bin/env python3
"""
Hypothesis: 1h RSI(14) mean reversion with 4h EMA50 trend filter and volume spike.
- Primary timeframe: 1h for execution, HTF: 4h for EMA50 trend filter.
- Mean reversion: RSI(14) < 30 for long, RSI(14) > 70 for short on 1h.
- Trend filter: Only trade in direction of 4h EMA50 (long if EMA50 rising, short if falling).
- Volume confirmation: current volume > 2.0x 20-period volume MA to ensure strong participation.
- Session filter: Only trade between 08:00-20:00 UTC to avoid low-liquidity hours.
- Discrete signal size: 0.20 to limit drawdown and reduce fee churn.
- Target: 60-150 total trades over 4 years (15-37/year) for 1h timeframe.
- Works in bull via buying oversold dips in uptrend, in bear via selling overbought rallies in downtrend.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Extract price and volume data
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Get 4h data for EMA50 trend filter
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 50:
        return np.zeros(n)
    
    close_4h = df_4h['close'].values
    
    # 4h EMA50 for trend filter
    ema_50_4h = pd.Series(close_4h).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_4h_aligned = align_htf_to_ltf(prices, df_4h, ema_50_4h)
    
    # 1h RSI(14) for mean reversion
    delta = pd.Series(close).diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(alpha=1/14, adjust=False, min_periods=14).mean()
    avg_loss = loss.ewm(alpha=1/14, adjust=False, min_periods=14).mean()
    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    rsi_values = rsi.values
    
    # Volume confirmation: current volume > 2.0 * 20-period volume MA
    volume_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (2.0 * volume_ma)
    
    # Session filter: 08:00-20:00 UTC
    hours = prices.index.hour  # open_time is already datetime64[ms]
    in_session = (hours >= 8) & (hours <= 20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start from index where all indicators are ready
    start_idx = max(50, 14, 20)  # EMA50 + RSI + volume MA
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(ema_50_4h_aligned[i]) or np.isnan(rsi_values[i]) or
            np.isnan(volume_spike[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Apply session filter
        if not in_session[i]:
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Only trade in direction of 4h EMA50 trend
            if i > 0 and not np.isnan(ema_50_4h_aligned[i-1]):
                ema50_slope = ema_50_4h_aligned[i] - ema_50_4h_aligned[i-1]
                if ema50_slope > 0:  # Uptrend
                    # Long when RSI < 30 (oversold) with volume spike
                    if rsi_values[i] < 30 and volume_spike[i]:
                        signals[i] = 0.20
                        position = 1
                elif ema50_slope < 0:  # Downtrend
                    # Short when RSI > 70 (overbought) with volume spike
                    if rsi_values[i] > 70 and volume_spike[i]:
                        signals[i] = -0.20
                        position = -1
        elif position == 1:
            # Long exit: RSI > 50 (mean reversion) or opposite signal
            if rsi_values[i] > 50:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.20
        elif position == -1:
            # Short exit: RSI < 50 (mean reversion) or opposite signal
            if rsi_values[i] < 50:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.20
    
    return signals

name = "1h_RSI14_MeanReversion_4hEMA50_Trend_VolumeSpike_v1"
timeframe = "1h"
leverage = 1.0
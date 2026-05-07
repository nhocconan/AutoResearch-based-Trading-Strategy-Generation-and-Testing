#!/usr/bin/env python3
"""
1h_RSI_4hTrend_1dVolume
Hypothesis: On 1h timeframe, use RSI(14) for mean-reversion entries aligned with 4h trend filter and 1d volume spike.
In uptrend (price > 4h EMA50), buy when RSI < 30 and volume > 1.5x 20-period average.
In downtrend (price < 4h EMA50), sell when RSI > 70 and volume > 1.5x 20-period average.
Exit when RSI crosses back to neutral (40 for longs, 60 for shorts) or trend flips.
Adds session filter (08-20 UTC) to avoid low-liquidity hours. Target: 20-40 trades per year (~80-160 over 4 years).
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "1h_RSI_4hTrend_1dVolume"
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
    
    # Pre-calc session filter (08-20 UTC)
    hours = prices.index.hour
    in_session = (hours >= 8) & (hours <= 20)
    
    # Load 4h data ONCE for trend filter
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 30:
        return np.zeros(n)
    
    # 4h EMA50 for trend filter
    ema_50_4h = pd.Series(df_4h['close']).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_4h_aligned = align_htf_to_ltf(prices, df_4h, ema_50_4h)
    
    # Load 1d data ONCE for volume filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    
    # 1d volume 20-period average
    vol_ma_1d = pd.Series(df_1d['volume']).rolling(window=20, min_periods=20).mean().values
    vol_ma_1d_aligned = align_htf_to_ltf(prices, df_1d, vol_ma_1d)
    
    # RSI(14) on 1h
    delta = np.diff(close, prepend=close[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    avg_gain = pd.Series(gain).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    avg_loss = pd.Series(loss).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    rs = np.where(avg_loss != 0, avg_gain / avg_loss, 100)
    rsi = 100 - (100 / (1 + rs))
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 30  # Need 30 periods for EMA50 and RSI warmup
    
    for i in range(start_idx, n):
        if not in_session[i]:
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if np.isnan(ema_50_4h_aligned[i]) or np.isnan(vol_ma_1d_aligned[i]) or np.isnan(rsi[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Determine market regime from 4h EMA50
        uptrend_regime = close[i] > ema_50_4h_aligned[i]
        downtrend_regime = close[i] < ema_50_4h_aligned[i]
        
        # Volume confirmation: current 1h volume > 1.5x 1d average volume
        # Note: Comparing 1h volume to daily average requires scaling
        vol_1h = volume[i]
        vol_ma_1h_equiv = vol_ma_1d_aligned[i] / 24.0  # Approximate hourly equivalent
        volume_confirm = vol_1h > 1.5 * vol_ma_1h_equiv if vol_ma_1h_equiv > 0 else False
        
        if position == 0:
            # Long: RSI < 30 (oversold) in uptrend + volume spike
            long_entry = (rsi[i] < 30) and uptrend_regime and volume_confirm
            # Short: RSI > 70 (overbought) in downtrend + volume spike
            short_entry = (rsi[i] > 70) and downtrend_regime and volume_confirm
            
            if long_entry:
                signals[i] = 0.20
                position = 1
            elif short_entry:
                signals[i] = -0.20
                position = -1
        elif position == 1:
            # Exit: RSI crosses above 40 or trend flips to downtrend
            if (rsi[i] > 40) or (not uptrend_regime):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.20
        elif position == -1:
            # Exit: RSI crosses below 60 or trend flips to uptrend
            if (rsi[i] < 60) or (not downtrend_regime):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.20
    
    return signals
#!/usr/bin/env python3
name = "1h_4h_1d_RSIVolumeTrend_v1"
timeframe = "1h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 200:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Load 4h data ONCE before loop for trend filter
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 50:
        return np.zeros(n)
    
    # Load daily data ONCE before loop for RSI and volume context
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # 4h EMA(50) for trend filter
    ema_50_4h = pd.Series(df_4h['close']).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_4h_aligned = align_htf_to_ltf(prices, df_4h, ema_50_4h)
    
    # Daily RSI(14) for momentum filter
    delta = pd.Series(df_1d['close']).diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(alpha=1/14, adjust=False, min_periods=14).mean()
    avg_loss = loss.ewm(alpha=1/14, adjust=False, min_periods=14).mean()
    rs = avg_gain / avg_loss
    rsi_14 = 100 - (100 / (1 + rs))
    rsi_14_vals = rsi_14.values
    rsi_14_aligned = align_htf_to_ltf(prices, df_1d, rsi_14_vals)
    
    # Daily volume SMA(20) for volume filter
    vol_sma_20 = pd.Series(df_1d['volume']).rolling(window=20, min_periods=20).mean().values
    vol_sma_20_aligned = align_htf_to_ltf(prices, df_1d, vol_sma_20)
    
    # Hourly session filter (08-20 UTC)
    hours = pd.DatetimeIndex(prices["open_time"]).hour
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 50  # Wait for EMA and RSI
    
    for i in range(start_idx, n):
        if (np.isnan(ema_50_4h_aligned[i]) or np.isnan(rsi_14_aligned[i]) or 
            np.isnan(vol_sma_20_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        hour = hours[i]
        in_session = (8 <= hour <= 20)
        
        if position == 0:
            # Long: RSI < 40 (oversold) + price > 4h EMA50 + volume > 1.5x daily avg + session
            vol_condition = volume[i] > vol_sma_20_aligned[i] * 1.5
            if (rsi_14_aligned[i] < 40 and 
                close[i] > ema_50_4h_aligned[i] and 
                vol_condition and 
                in_session):
                signals[i] = 0.20
                position = 1
            # Short: RSI > 60 (overbought) + price < 4h EMA50 + volume > 1.5x daily avg + session
            elif (rsi_14_aligned[i] > 60 and 
                  close[i] < ema_50_4h_aligned[i] and 
                  vol_condition and 
                  in_session):
                signals[i] = -0.20
                position = -1
        elif position == 1:
            # Exit: RSI > 50 (momentum fading) or price < 4h EMA50 or outside session
            if (rsi_14_aligned[i] > 50 or 
                close[i] < ema_50_4h_aligned[i] or 
                not in_session):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.20
        elif position == -1:
            # Exit: RSI < 50 (momentum fading) or price > 4h EMA50 or outside session
            if (rsi_14_aligned[i] < 50 or 
                close[i] > ema_50_4h_aligned[i] or 
                not in_session):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.20
    
    return signals

# Hypothesis: 1h RSI mean reversion with 4h trend filter and volume confirmation
# - Uses daily RSI(14) for oversold/overbought signals (<40/>60)
# - 4h EMA(50) filter ensures trades align with higher timeframe trend
# - Volume spike (>1.5x daily average) confirms institutional participation
# - Session filter (08-20 UTC) reduces noise during low-liquidity hours
# - Works in both bull and bear markets: buys oversold dips in uptrend, sells overbought rallies in downtrend
# - Position size 0.20 limits drawdown while allowing meaningful returns
# - Target: 15-35 trades/year to avoid fee drag (60-140 total over 4 years)
# - Exits when RSI reverts to neutral (50) or trend fails or outside session
# - Novel combination: daily RSI + 4h trend + volume filter not recently tried on 1h timeframe
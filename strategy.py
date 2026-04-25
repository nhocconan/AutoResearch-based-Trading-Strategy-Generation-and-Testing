#!/usr/bin/env python3
"""
1h Volume Spike + 4h RSI(14) Extreme + 1d EMA50 Trend Filter
Hypothesis: On 1h timeframe, volume spikes combined with 4h RSI extremes (oversold/overbought) 
provide high-probability mean-reversion entries when aligned with the 1d EMA50 trend. 
The strategy uses the 4h RSI for signal direction (avoiding 1h noise) and 1h only for precise 
entry timing via volume spikes. Daily EMA50 filter ensures we trade with the higher timeframe 
trend, working in both bull (trend continuation on pullbacks) and bear (mean reversion in 
range-bound markets) conditions. Target: 15-30 trades/year (60-120 over 4 years) to minimize 
fee drag on this difficult timeframe. Uses discrete position sizing of 0.20 to control 
drawdown and reduce signal churn.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def calculate_rsi(series, period):
    """Calculate Relative Strength Index"""
    if len(series) < period:
        return np.full_like(series, np.nan)
    delta = pd.Series(series).diff()
    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)
    avg_gain = gain.ewm(alpha=1/period, adjust=False, min_periods=period).mean()
    avg_loss = loss.ewm(alpha=1/period, adjust=False, min_periods=period).mean()
    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    return rsi.values

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # 4h data for RSI (loaded ONCE)
    df_4h = get_htf_data(prices, '4h')
    rsi_4h = calculate_rsi(df_4h['close'].values, 14)
    rsi_4h_aligned = align_htf_to_ltf(prices, df_4h, rsi_4h)
    
    # 1d data for EMA50 trend (loaded ONCE)
    df_1d = get_htf_data(prices, '1d')
    ema_50_1d = pd.Series(df_1d['close'].values).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # 1h volume spike: current volume > 2.5 * 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (vol_ma * 2.5)
    
    # Session filter: 08-20 UTC (pre-compute hours for efficiency)
    hours = pd.DatetimeIndex(prices["open_time"]).hour
    in_session = (hours >= 8) & (hours <= 20)
    
    signals = np.zeros(n)
    
    # Start index: need enough for all indicators
    start_idx = max(50, 20) + 5
    
    for i in range(start_idx, n):
        # Skip if outside trading session
        if not in_session[i]:
            signals[i] = 0.0
            continue
            
        # Skip if any data not ready
        if (np.isnan(rsi_4h_aligned[i]) or np.isnan(ema_50_1d_aligned[i]) or 
            np.isnan(vol_ma[i])):
            signals[i] = 0.0
            continue
        
        curr_close = close[i]
        vol_spike = volume_spike[i]
        
        # Mean reversion conditions: RSI extreme + volume spike
        rsi_oversold = rsi_4h_aligned[i] < 30
        rsi_overbought = rsi_4h_aligned[i] > 70
        
        # Trend alignment: price relative to daily EMA50
        above_ema = curr_close > ema_50_1d_aligned[i]
        below_ema = curr_close < ema_50_1d_aligned[i]
        
        # Entry logic: fade extremes with volume confirmation
        long_entry = rsi_oversold and vol_spike and below_ema
        short_entry = rsi_overbought and vol_spike and above_ema
        
        if long_entry:
            signals[i] = 0.20
        elif short_entry:
            signals[i] = -0.20
        else:
            signals[i] = 0.0
    
    return signals

name = "1h_VolumeSpike_RSI4hExtreme_EMA50dTrend"
timeframe = "1h"
leverage = 1.0
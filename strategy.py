#!/usr/bin/env python3
"""
12h Volume Spike + 1d EMA50 Trend Reversal - Contrarian entries at extreme volume spikes 
with daily EMA50 trend filter. Uses ATR-based stoploss for risk control.
Hypothesis: Extreme volume spikes often mark exhaustion points. In bull markets, 
spikes during uptrend continuation can signal pullbacks to buy. In bear markets, 
spikes during downtrend can signal bounces to sell. Daily EMA50 filter ensures we 
trade with the higher timeframe trend. Target: 50-150 trades over 4 years.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def calculate_ema(series, period):
    """Calculate Exponential Moving Average"""
    if len(series) < period:
        return np.full_like(series, np.nan)
    return pd.Series(series).ewm(span=period, adjust=False, min_periods=period).mean().values

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    open_ = prices['open'].values
    
    # Daily data for EMA50 trend (loaded ONCE)
    df_1d = get_htf_data(prices, '1d')
    
    # Daily EMA50 trend filter
    ema_50_1d = calculate_ema(df_1d['close'].values, 50)
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # Volume confirmation: current volume > 3.0 * 20-period average (extreme spike only)
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (vol_ma * 3.0)
    
    # ATR for stoploss (14-period)
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]  # First value
    atr = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    # Price position relative to 12-period 12h EMA for entry timing
    ema_12_12h = calculate_ema(close, 12)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    # Start index: need enough for EMA and volume MA
    start_idx = max(50, 20, 14, 12) + 5
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(ema_50_1d_aligned[i]) or np.isnan(vol_ma[i]) or np.isnan(atr[i]) or
            np.isnan(ema_12_12h[i])):
            signals[i] = 0.0
            continue
        
        curr_close = close[i]
        vol_spike = volume_spike[i]
        
        if position == 0:
            # Look for entry signals - require: extreme volume spike + price vs 12h EMA + daily trend alignment
            # Long: volume spike + price below 12h EMA (pullback in uptrend) + price above daily EMA50
            long_entry = vol_spike and (curr_close < ema_12_12h[i]) and (curr_close > ema_50_1d_aligned[i])
            # Short: volume spike + price above 12h EMA (bounce in downtrend) + price below daily EMA50
            short_entry = vol_spike and (curr_close > ema_12_12h[i]) and (curr_close < ema_50_1d_aligned[i])
            
            if long_entry:
                signals[i] = 0.25
                position = 1
                entry_price = curr_close
            elif short_entry:
                signals[i] = -0.25
                position = -1
                entry_price = curr_close
            else:
                signals[i] = 0.0
        elif position == 1:
            # Long position: exit on retrace to 12h EMA, trend change, or ATR stoploss
            stoploss_level = entry_price - 2.5 * atr[i]
            if curr_close > ema_12_12h[i] or curr_close < ema_50_1d_aligned[i] or curr_close < stoploss_level:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short position: exit on retrace to 12h EMA, trend change, or ATR stoploss
            stoploss_level = entry_price + 2.5 * atr[i]
            if curr_close < ema_12_12h[i] or curr_close > ema_50_1d_aligned[i] or curr_close > stoploss_level:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "12h_VolumeSpike_EMA50Trend_Reversal_ATRStop"
timeframe = "12h"
leverage = 1.0
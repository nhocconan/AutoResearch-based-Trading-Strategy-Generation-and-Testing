#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Williams %R mean reversion with 1d trend filter and volume confirmation
# - Williams %R(14) < -80 = oversold (long setup), > -20 = overbought (short setup)
# - Requires 1d EMA50 trend alignment: long only when 1d close > EMA50, short only when < EMA50
# - Volume confirmation: current volume > 1.5x 20-bar average to avoid low-liquidity false signals
# - Exit when Williams %R reverts to -50 (mean reversion target) or trend fails
# - Designed for 6h timeframe to capture swing reversals in both bull and bear markets
# - Target: 50-120 total trades over 4 years (12-30/year) to minimize fee drag
# - Size: 0.25 (25% of capital per trade)

name = "6h_1d_williamsr_meanreversion_volume_trend_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Load HTF data ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # Pre-compute Williams %R(14) on 6h data
    highest_high = prices['high'].rolling(window=14, min_periods=14).max()
    lowest_low = prices['low'].rolling(window=14, min_periods=14).min()
    williams_r = -100 * (highest_high - prices['close']) / (highest_high - lowest_low)
    williams_r = williams_r.replace([np.inf, -np.inf], np.nan).values
    
    # Pre-compute volume confirmation: > 1.5x 20-period average
    volume_20_avg = prices['volume'].rolling(window=20, min_periods=20).mean()
    vol_spike = (prices['volume'] > (1.5 * volume_20_avg)).values
    
    # Pre-compute aligned 1d data
    h_1d = df_1d['high'].values
    l_1d = df_1d['low'].values
    c_1d = df_1d['close'].values
    
    h_1d_aligned = align_htf_to_ltf(prices, df_1d, h_1d)
    l_1d_aligned = align_htf_to_ltf(prices, df_1d, l_1d)
    c_1d_aligned = align_htf_to_ltf(prices, df_1d, c_1d)
    
    # Pre-compute 1d EMA(50) for trend filter
    ema50_1d = pd.Series(c_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema50_1d)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(20, n):  # Start after warmup period
        # Skip if any required data is invalid
        if (np.isnan(williams_r[i]) or np.isnan(volume_20_avg.iloc[i]) or 
            np.isnan(ema50_1d_aligned[i]) or np.isnan(c_1d_aligned[i])):
            # Hold current position or flat
            if position == 0:
                signals[i] = 0.0
            elif position == 1:
                signals[i] = 0.25
            else:
                signals[i] = -0.25
            continue
        
        # Get current values
        wr = williams_r[i]
        close_price = prices['close'].iloc[i]
        vol_ok = vol_spike[i]
        ema_trend = ema50_1d_aligned[i]
        current_1d_close = c_1d_aligned[i]
        
        if position == 0:  # Flat - look for new entries
            # Long setup: oversold + uptrend + volume confirmation
            if (wr < -80 and 
                current_1d_close > ema_trend and 
                vol_ok):
                position = 1
                signals[i] = 0.25
            # Short setup: overbought + downtrend + volume confirmation
            elif (wr > -20 and 
                  current_1d_close < ema_trend and 
                  vol_ok):
                position = -1
                signals[i] = -0.25
        
        elif position == 1:  # Long position - look for exit
            # Exit conditions:
            # 1. Williams %R reverts to -50 (mean reversion target)
            # 2. Trend fails (1d close < EMA50)
            if (wr >= -50 or current_1d_close < ema_trend):
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25  # Hold long
        
        elif position == -1:  # Short position - look for exit
            # Exit conditions:
            # 1. Williams %R reverts to -50 (mean reversion target)
            # 2. Trend fails (1d close > EMA50)
            if (wr <= -50 or current_1d_close > ema_trend):
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25  # Hold short
    
    return signals
#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h TD Sequential Setup with 1d Trend Filter and Volume Confirmation
# - TD Sequential identifies exhaustion points (setup 9) for mean reversion
# - Uses 1d EMA50 trend filter to align with higher timeframe direction
# - Volume spike confirms the setup validity
# - Works in both bull and bear by trading reversals within the trend
# - Target: 20-40 trades/year to minimize fee drag on 4h timeframe

name = "4h_TDSequential_1dTrend_Volume"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # 1d data for trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    # 1d EMA50 for trend filter
    ema_50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # TD Sequential setup (simplified: count consecutive closes >/< 4 periods ago)
    # Buy setup: close > close 4 periods ago for 9 consecutive periods
    # Sell setup: close < close 4 periods ago for 9 consecutive periods
    td_buy_setup = np.zeros(n, dtype=int)
    td_sell_setup = np.zeros(n, dtype=int)
    
    buy_count = 0
    sell_count = 0
    
    for i in range(4, n):
        if close[i] > close[i-4]:
            buy_count += 1
            sell_count = 0
        elif close[i] < close[i-4]:
            sell_count += 1
            buy_count = 0
        else:
            buy_count = 0
            sell_count = 0
        
        td_buy_setup[i] = buy_count
        td_sell_setup[i] = sell_count
    
    # Volume spike: current volume > 2.0x 20-period average
    vol_ma20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (2.0 * vol_ma20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 100
    
    for i in range(start_idx, n):
        # Skip if any critical data is NaN
        if (np.isnan(ema_50_1d_aligned[i]) or np.isnan(volume_spike[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: TD sell setup >= 9 (exhaustion) with 1d uptrend + volume spike
            long_cond = (td_sell_setup[i] >= 9 and 
                        ema_50_1d_aligned[i] > ema_50_1d_aligned[i-1] and
                        volume_spike[i])
            
            # Short: TD buy setup >= 9 (exhaustion) with 1d downtrend + volume spike
            short_cond = (td_buy_setup[i] >= 9 and 
                         ema_50_1d_aligned[i] < ema_50_1d_aligned[i-1] and
                         volume_spike[i])
            
            if long_cond:
                signals[i] = 0.25
                position = 1
            elif short_cond:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Long exit: TD buy setup >= 9 (new exhaustion) or trend change
            if (td_buy_setup[i] >= 9 or 
                ema_50_1d_aligned[i] < ema_50_1d_aligned[i-1]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: TD sell setup >= 9 (new exhaustion) or trend change
            if (td_sell_setup[i] >= 9 or 
                ema_50_1d_aligned[i] > ema_50_1d_aligned[i-1]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals
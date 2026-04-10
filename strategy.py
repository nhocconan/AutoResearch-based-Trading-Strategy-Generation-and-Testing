#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Donchian(20) breakout with 1d ADX trend filter and volume confirmation
# - Long when price breaks above Donchian(20) high with volume > 1.5x 20-bar avg AND 1d ADX > 25
# - Short when price breaks below Donchian(20) low with volume > 1.5x 20-bar avg AND 1d ADX > 25
# - Exit when price retraces to Donchian midpoint OR ATR-based stoploss (2.5x ATR)
# - Uses 1d ADX to filter for trending markets only, avoiding choppy conditions
# - Discrete position sizing (0.25) to minimize fee churn
# - Target: 20-35 trades/year on 4h timeframe (80-140 total over 4 years)
# - Focus on BTC/ETH; SOL-only strategies are low value

name = "4h_1d_donchian_breakout_adx_volume_v1"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Load HTF data ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # Pre-compute Donchian channels (20-period)
    donchian_high = prices['high'].rolling(window=20, min_periods=20).max().values
    donchian_low = prices['low'].rolling(window=20, min_periods=20).min().values
    donchian_mid = (donchian_high + donchian_low) / 2.0
    
    # Pre-compute volume confirmation: > 1.5x 20-period average
    volume_20_avg = prices['volume'].rolling(window=20, min_periods=20).mean().values
    vol_spike = prices['volume'] > (1.5 * volume_20_avg)
    
    # Pre-compute ATR(14) for stoploss
    high_low = prices['high'] - prices['low']
    high_close = np.abs(prices['high'] - prices['close'].shift(1))
    low_close = np.abs(prices['low'] - prices['close'].shift(1))
    tr = np.maximum(np.maximum(high_low, high_close), low_close)
    atr = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    entry_price = 0.0  # track entry price for stoploss
    
    # Pre-compute aligned 1d data for ADX calculation
    h_1d = df_1d['high'].values
    l_1d = df_1d['low'].values
    c_1d = df_1d['close'].values
    
    h_1d_aligned = align_htf_to_ltf(prices, df_1d, h_1d)
    l_1d_aligned = align_htf_to_ltf(prices, df_1d, l_1d)
    c_1d_aligned = align_htf_to_ltf(prices, df_1d, c_1d)
    
    # Pre-compute 1d ADX(14) for trend filter
    # Calculate +DM, -DM, TR
    plus_dm = np.zeros_like(h_1d)
    minus_dm = np.zeros_like(h_1d)
    tr_1d = np.zeros_like(h_1d)
    
    for i in range(1, len(h_1d)):
        plus_dm[i] = max(0, h_1d[i] - h_1d[i-1])
        minus_dm[i] = max(0, l_1d[i-1] - l_1d[i])
        tr_1d[i] = max(h_1d[i] - l_1d[i], abs(h_1d[i] - c_1d[i-1]), abs(l_1d[i] - c_1d[i-1]))
        
        # Fix for DM: if +DM > -DM then -DM=0, if -DM > +DM then +DM=0
        if plus_dm[i] > minus_dm[i]:
            minus_dm[i] = 0
        elif minus_dm[i] > plus_dm[i]:
            plus_dm[i] = 0
        else:
            plus_dm[i] = 0
            minus_dm[i] = 0
    
    # Smooth with Wilder's smoothing (alpha = 1/period)
    def wilders_smoothing(data, period):
        result = np.full_like(data, np.nan)
        if len(data) < period:
            return result
        # First value is simple average
        result[period-1] = np.mean(data[1:period])
        # Subsequent values: Wilder's smoothing
        for i in range(period, len(data)):
            result[i] = (result[i-1] * (period-1) + data[i]) / period
        return result
    
    period = 14
    smoothed_plus_dm = wilders_smoothing(plus_dm, period)
    smoothed_minus_dm = wilders_smoothing(minus_dm, period)
    smoothed_tr = wilders_smoothing(tr_1d, period)
    
    # Calculate +DI and -DI
    plus_di = 100 * smoothed_plus_dm / smoothed_tr
    minus_di = 100 * smoothed_minus_dm / smoothed_tr
    
    # Calculate DX and ADX
    dx = np.full_like(plus_di, np.nan)
    dx_denom = plus_di + minus_di
    valid = dx_denom > 0
    dx[valid] = 100 * np.abs(plus_di[valid] - minus_di[valid]) / dx_denom[valid]
    
    adx = wilders_smoothing(dx, period)
    adx_aligned = align_htf_to_ltf(prices, df_1d, adx)
    
    for i in range(20, n):
        # Skip if any required data is invalid
        if (np.isnan(donchian_high[i]) or np.isnan(donchian_low[i]) or 
            np.isnan(volume_20_avg[i]) or np.isnan(atr[i]) or 
            np.isnan(adx_aligned[i])):
            # Hold current position or flat
            if position == 0:
                signals[i] = 0.0
            elif position == 1:
                signals[i] = 0.25
            else:
                signals[i] = -0.25
            continue
        
        if position == 0:  # Flat - look for new breakout entries
            # Long breakout: price > Donchian high with volume spike AND 1d ADX > 25 (trending)
            if (prices['close'].iloc[i] > donchian_high[i] and 
                vol_spike.iloc[i] and 
                adx_aligned[i] > 25):
                position = 1
                entry_price = prices['close'].iloc[i]
                signals[i] = 0.25
            # Short breakdown: price < Donchian low with volume spike AND 1d ADX > 25 (trending)
            elif (prices['close'].iloc[i] < donchian_low[i] and 
                  vol_spike.iloc[i] and 
                  adx_aligned[i] > 25):
                position = -1
                entry_price = prices['close'].iloc[i]
                signals[i] = -0.25
            else:
                signals[i] = 0.0
        else:  # Have position - look for exit
            # Exit conditions:
            # 1. Price retraces to Donchian midpoint
            # 2. ATR-based stoploss hit
            exit_signal = False
            if position == 1:  # Long position
                if (prices['close'].iloc[i] < donchian_mid[i] or
                    prices['close'].iloc[i] < entry_price - 2.5 * atr[i]):
                    exit_signal = True
            elif position == -1:  # Short position
                if (prices['close'].iloc[i] > donchian_mid[i] or
                    prices['close'].iloc[i] > entry_price + 2.5 * atr[i]):
                    exit_signal = True
            
            if exit_signal:
                position = 0
                entry_price = 0.0
                signals[i] = 0.0
            else:
                if position == 1:
                    signals[i] = 0.25
                else:
                    signals[i] = -0.25
    
    return signals
#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1h VWAP Mean Reversion with 4h Trend Filter and Session Filter
# Uses 1h VWAP deviation for mean reversion signals - price tends to revert to VWAP
# 4h EMA (50) provides trend filter to avoid counter-trend trades
# Session filter (08-20 UTC) reduces noise during low-volume periods
# Target: 15-37 trades/year (60-150 total over 4 years) to minimize fee drag

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Load 4h data ONCE before loop for trend filter
    df_4h = get_htf_data(prices, '4h')
    
    # Calculate 4h EMA (50) for trend direction
    close_4h = df_4h['close'].values
    ema_4h = pd.Series(close_4h).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_4h_aligned = align_htf_to_ltf(prices, df_4h, ema_4h)
    
    # Calculate VWAP (20-period) on 1h data
    typical_price = (high + low + close) / 3.0
    tpv = typical_price * volume
    cum_tpv = np.nancumsum(tpv)  # cumulative sum treating NaN as 0
    cum_vol = np.nancumsum(volume)
    vwap = np.full_like(typical_price, np.nan)
    valid_vol = cum_vol != 0
    vwap[valid_vol] = cum_tpv[valid_vol] / cum_vol[valid_vol]
    
    # Standard deviation of price from VWAP (20-period)
    price_dev = typical_price - vwap
    dev_series = pd.Series(price_dev)
    std_dev = dev_series.rolling(window=20, min_periods=20).std().values
    
    signals = np.zeros(n)
    position = 0
    position_size = 0.20  # 20% position size
    
    # Start after enough data for calculations
    start = 20  # for VWAP and std dev
    
    for i in range(start, n):
        # Skip if any critical data is NaN
        if (np.isnan(vwap[i]) or np.isnan(std_dev[i]) or 
            np.isnan(ema_4h_aligned[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        
        # Trend filter: only trade in direction of 4h EMA
        above_ema = price > ema_4h_aligned[i]
        
        if position == 0:
            # Long: price reverts to VWAP from below (oversold) with uptrend filter
            if price < vwap[i] - 1.5 * std_dev[i] and above_ema:
                position = 1
                signals[i] = position_size
            # Short: price reverts to VWAP from above (overbought) with downtrend filter
            elif price > vwap[i] + 1.5 * std_dev[i] and not above_ema:
                position = -1
                signals[i] = -position_size
            else:
                signals[i] = 0.0
        elif position == 1:
            # Exit long: price reaches VWAP (mean reversion complete) or trend changes
            if price >= vwap[i] or price < ema_4h_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = position_size
        elif position == -1:
            # Exit short: price reaches VWAP (mean reversion complete) or trend changes
            if price <= vwap[i] or price > ema_4h_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -position_size
    
    return signals

name = "1h_VWAP_MeanReversion_4hEMA_Trend"
timeframe = "1h"
leverage = 1.0
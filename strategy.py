# 12h Volume-Weighted VWAP Reversion with 1-week Trend Filter
# Hypothesis: Mean reversion to VWAP on 12h timeframe with weekly trend filter works in both bull and bear markets.
# Uses weekly EMA for trend direction and 12h VWAP deviation for mean reversion signals.
# Volume-weighted entry ensures institutional participation. Target: 15-35 trades/year to avoid fee drag.

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    open_price = prices['open'].values
    
    # Get weekly data for trend filter
    df_1w = get_htf_data(prices, '1w')
    close_1w = df_1w['close'].values
    
    # Calculate weekly EMA(34) for trend
    ema_34_1w = pd.Series(close_1w).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_34_1w)
    
    # Calculate typical price and VWAP components
    typical_price = (high + low + close) / 3.0
    vwap_num = typical_price * volume
    vwap_den = volume
    
    # Calculate cumulative VWAP for each 12h period (reset daily)
    # We'll calculate VWAP from session start (simplified: use cumulative since start of data)
    cum_vwap_num = np.cumsum(vwap_num)
    cum_vwap_den = np.cumsum(vwap_den)
    vwap = cum_vwap_num / cum_vwap_den
    
    # Calculate 12-period standard deviation of price deviation from VWAP
    price_dev = typical_price - vwap
    # Use rolling window for volatility normalization
    vol_lookback = 12
    vwap_std = np.full(n, np.nan)
    for i in range(vol_lookback, n):
        vwap_std[i] = np.std(price_dev[i-vol_lookback:i])
    
    # Avoid division by zero
    vwap_std[vwap_std == 0] = 1e-10
    
    # Normalized deviation from VWAP (Z-score)
    vwap_zscore = price_dev / vwap_std
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(34, vol_lookback)  # need weekly EMA and VWAP std
    
    for i in range(start_idx, n):
        # Skip if any required data is not available
        if np.isnan(ema_34_1w_aligned[i]) or np.isnan(vwap_std[i]) or vwap_std[i] < 1e-10:
            signals[i] = 0.0
            continue
        
        zscore = vwap_zscore[i]
        price_above_vwap = typical_price[i] > vwap[i]
        price_below_vwap = typical_price[i] < vwap[i]
        
        # Trend filter: weekly EMA slope
        if i > start_idx:
            ema_now = ema_34_1w_aligned[i]
            ema_prev = ema_34_1w_aligned[i-1]
            trend_up = ema_now > ema_prev
            trend_down = ema_now < ema_prev
        else:
            trend_up = True
            trend_down = True
        
        if position == 0:
            # Long entry: price significantly below VWAP in uptrend
            if (zscore < -1.5 and  # price below VWAP
                trend_up):         # weekly uptrend
                signals[i] = 0.25
                position = 1
            # Short entry: price significantly above VWAP in downtrend
            elif (zscore > 1.5 and   # price above VWAP
                  trend_down):       # weekly downtrend
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        
        elif position == 1:
            # Long exit: price crosses back above VWAP or opposite signal
            if zscore > 0.5:  # return toward VWAP
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: price crosses back below VWAP or opposite signal
            if zscore < -0.5:  # return toward VWAP
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "12h_VWAPReversion_WeeklyEMA34_Trend"
timeframe = "12h"
leverage = 1.0
#150151
#!/usr/bin/env python3
# 6h_Session_VWAP_Deviation_1dTrend
# Hypothesis: Mean-revert from session VWAP with 1d trend filter and volume confirmation.
# In 6h timeframe, price often deviates from daily VWAP then reverts, especially in ranging markets.
# Long when price < VWAP - 1*sigma and in 1d uptrend with volume spike.
# Short when price > VWAP + 1*sigma and in 1d downtrend with volume spike.
# Uses 6h session VWAP (reset daily) and standard deviation bands.
# Target: 50-150 total trades over 4 years with disciplined entries to avoid fee drag.

name = "6h_Session_VWAP_Deviation_1dTrend"
timeframe = "6h"
leverage = 1.0

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
    open_time = pd.to_datetime(prices['open_time'])
    
    # Get daily data for trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    # Daily EMA for trend filter
    ema_34_1d = pd.Series(df_1d['close']).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # Calculate session VWAP and std dev (reset daily)
    typical_price = (high + low + close) / 3.0
    vwap = np.full(n, np.nan)
    vwap_sum = np.full(n, np.nan)
    vol_sum = np.full(n, np.nan)
    
    # Track daily session
    current_date = None
    session_tp_vol = 0.0
    session_vol = 0.0
    
    for i in range(n):
        date = open_time[i].date()
        if date != current_date:
            # New session, reset accumulators
            current_date = date
            session_tp_vol = 0.0
            session_vol = 0.0
        
        session_tp_vol += typical_price[i] * volume[i]
        session_vol += volume[i]
        
        if session_vol > 0:
            vwap[i] = session_tp_vol / session_vol
    
    # Calculate rolling standard deviation of price-VWAP deviation
    price_dev = typical_price - vwap
    # Use 20-period rolling std dev of the deviation
    price_dev_series = pd.Series(price_dev)
    std_dev = price_dev_series.rolling(window=20, min_periods=20).std().values
    
    # Volume confirmation (20-period MA)
    volume_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(34, 20)  # Warmup for daily EMA and VWAP/std calculations
    
    for i in range(start_idx, n):
        # Skip if any critical values are NaN
        if (np.isnan(vwap[i]) or np.isnan(std_dev[i]) or 
            np.isnan(ema_34_1d_aligned[i]) or np.isnan(volume_ma[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Daily trend filter
        uptrend = close[i] > ema_34_1d_aligned[i]
        downtrend = close[i] < ema_34_1d_aligned[i]
        
        # Volume confirmation
        volume_confirm = volume[i] > volume_ma[i] * 1.5
        
        # Price deviation from VWAP
        if std_dev[i] > 0:
            dev_ratio = (typical_price[i] - vwap[i]) / std_dev[i]
        else:
            dev_ratio = 0
        
        if position == 0:
            # Long entry: price below VWAP by 1*sigma, uptrend, volume spike
            if dev_ratio < -1.0 and uptrend and volume_confirm:
                signals[i] = 0.25
                position = 1
            # Short entry: price above VWAP by 1*sigma, downtrend, volume spike
            elif dev_ratio > 1.0 and downtrend and volume_confirm:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Long exit: price crosses back above VWAP or trend breaks
            if typical_price[i] >= vwap[i] or not uptrend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: price crosses back below VWAP or trend breaks
            if typical_price[i] <= vwap[i] or not downtrend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals
#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Volume-Weighted Average Price (VWAP) Deviation + 1w Trend Filter + Volume Confirmation
# VWAP acts as a dynamic support/resistance level. Price deviations from VWAP combined with 
# weekly trend filter and volume spikes capture mean reversion in the direction of the higher 
# timeframe trend. Works in both bull and bear markets by aligning with weekly trend. 
# Target: 50-150 trades over 4 years (12-37/year) on 6h.

name = "6h_VWAP_Deviation_1wTrend_Volume"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Calculate 1w EMA50 for trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 50:
        return np.zeros(n)
    
    close_1w = df_1w['close'].values
    ema_50_1w = pd.Series(close_1w).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_50_1w)
    
    # Calculate VWAP (session-based: reset daily)
    typical_price = (high + low + close) / 3.0
    tp_volume = typical_price * volume
    
    # Get unique dates for daily reset
    dates = pd.to_datetime(prices['open_time']).date
    unique_dates = np.unique(dates)
    vwap = np.full(n, np.nan)
    
    for date in unique_dates:
        mask = (dates == date)
        if not np.any(mask):
            continue
        cum_tp_volume = np.nancumsum(tp_volume * mask)
        cum_volume = np.nancumsum(volume * mask)
        # Avoid division by zero
        vwap[mask] = np.divide(cum_tp_volume, cum_volume, 
                               out=np.full_like(cum_tp_volume, np.nan), 
                               where=(cum_volume != 0))
    
    # Calculate price deviation from VWAP (normalized by ATR-like measure)
    price_dev = (close - vwap) / vwap * 100  # Percentage deviation
    
    # Volume confirmation: 2.0x 20-period average (~8.3 days for 6h)
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (2.0 * vol_ma)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after warmup (need enough data for VWAP and 1w EMA)
    start_idx = max(50, 20)  # 1w EMA50 warmup
    
    for i in range(start_idx, n):
        # Check for NaN values in indicators
        if (np.isnan(ema_50_1w_aligned[i]) or np.isnan(price_dev[i]) or 
            np.isnan(vwap[i]) or np.isnan(vol_ma[i])):
            signals[i] = 0.0
            continue
        
        if position == 0:  # Flat - look for new entries
            # Long entry: Price significantly below VWAP (<-1.5%) with volume spike AND price > 1w EMA50 (bullish trend)
            if (price_dev[i] < -1.5 and 
                volume_spike[i] and 
                close[i] > ema_50_1w_aligned[i]):
                signals[i] = 0.25
                position = 1
            # Short entry: Price significantly above VWAP (>1.5%) with volume spike AND price < 1w EMA50 (bearish trend)
            elif (price_dev[i] > 1.5 and 
                  volume_spike[i] and 
                  close[i] < ema_50_1w_aligned[i]):
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        
        elif position == 1:  # Long position
            # Exit: Price crosses above VWAP (mean reversion complete) OR price below 1w EMA50 (trend change)
            if close[i] > vwap[i] or close[i] < ema_50_1w_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:  # Short position
            # Exit: Price crosses below VWAP (mean reversion complete) OR price above 1w EMA50 (trend change)
            if close[i] < vwap[i] or close[i] > ema_50_1w_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals
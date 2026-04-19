#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h timeframe with 1-day VWAP deviation and 1-week trend filter.
# Enters when price deviates significantly from 1-day VWAP (>1.5*ATR) in direction of weekly trend.
# Uses VWAP mean reversion in ranging markets and trend alignment to avoid counter-trend trades.
# Targets 15-25 trades/year (60-100 over 4 years) to minimize fee drag.
name = "6h_1d_VWAP_Deviation_1w_Trend"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    open_time = prices['open_time']
    
    # Pre-compute session filter (08-20 UTC)
    hours = pd.DatetimeIndex(open_time).hour
    session_filter = (hours >= 8) & (hours <= 20)
    
    # Get 1d data for VWAP calculation (called ONCE before loop)
    df_1d = get_htf_data(prices, '1d')
    typical_price_1d = (df_1d['high'].values + df_1d['low'].values + df_1d['close'].values) / 3.0
    volume_1d = df_1d['volume'].values
    vwap_1d = (np.cumsum(typical_price_1d * volume_1d) / np.cumsum(volume_1d))
    vwap_1d = np.where(np.cumsum(volume_1d) == 0, np.nan, vwap_1d)
    vwap_1d_aligned = align_htf_to_ltf(prices, df_1d, vwap_1d)
    
    # Get 1w data for EMA34 trend (called ONCE before loop)
    df_1w = get_htf_data(prices, '1w')
    close_1w = df_1w['close'].values
    ema_34_1w = pd.Series(close_1w).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_34_1w)
    
    # ATR for volatility normalization (14-period)
    tr1 = high[1:] - low[1:]
    tr2 = np.abs(high[1:] - close[:-1])
    tr3 = np.abs(low[1:] - close[:-1])
    tr = np.concatenate([[np.nan], np.maximum(tr1, np.maximum(tr2, tr3))])
    atr = pd.Series(tr).ewm(span=14, adjust=False, min_periods=14).mean().values
    
    # VWAP deviation in ATR units
    vwap_dev = (close - vwap_1d_aligned) / atr
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 50  # Ensure enough data for all indicators
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN or outside session
        if (np.isnan(vwap_dev[i]) or np.isnan(ema_34_1w_aligned[i]) or 
            np.isnan(vwap_1d_aligned[i]) or not session_filter[i]):
            signals[i] = 0.0
            continue
        
        if position == 0:
            # Long: price below VWAP (oversold) AND above weekly EMA34 (uptrend)
            if (vwap_dev[i] < -1.5 and 
                close[i] > ema_34_1w_aligned[i]):
                signals[i] = 0.25
                position = 1
            # Short: price above VWAP (overbought) AND below weekly EMA34 (downtrend)
            elif (vwap_dev[i] > 1.5 and 
                  close[i] < ema_34_1w_aligned[i]):
                signals[i] = -0.25
                position = -1
                
        elif position == 1:
            # Long: exit if price returns to VWAP or breaks below weekly EMA34
            if vwap_dev[i] > -0.5 or close[i] < ema_34_1w_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
                
        elif position == -1:
            # Short: exit if price returns to VWAP or breaks above weekly EMA34
            if vwap_dev[i] < 0.5 or close[i] > ema_34_1w_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals
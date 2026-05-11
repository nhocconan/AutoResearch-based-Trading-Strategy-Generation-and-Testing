#!/usr/bin/env python3
"""
6h_MonthlyVWAP_Deviation_1dTrend_Volume
Hypothesis: Price deviating from monthly VWAP (1-month rolling VWAP) indicates overextension, with mean reversion trades taken when price returns toward VWAP, filtered by 1d EMA34 trend and volume spike. Monthly VWAP adapts to longer-term value area, providing dynamic support/resistance. Works in bull (buy dips to VWAP in uptrend) and bear (sell rallies to VWAP in downtrend). Target: 15-30 trades/year to minimize fee drag.
"""

name = "6h_MonthlyVWAP_Deviation_1dTrend_Volume"
timeframe = "6h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Get 1d data for trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    # 6h OHLCV
    close_6h = prices['close'].values
    high_6h = prices['high'].values
    low_6h = prices['low'].values
    volume_6h = prices['volume'].values
    
    # --- 1d Trend Filter: EMA34 ---
    close_1d = df_1d['close'].values
    ema34_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema34_1d)
    
    # --- Monthly VWAP (approx 20 days * 4 periods per day = 80 periods for 6h) ---
    # Typical price
    typical_price = (high_6h + low_6h + close_6h) / 3.0
    # VWAP components
    pv = typical_price * volume_6h
    # Cumulative sums for VWAP
    cum_pv = np.nancumsum(pv)
    cum_vol = np.nancumsum(volume_6h)
    # Avoid division by zero
    vwap = np.divide(cum_pv, cum_vol, out=np.full_like(cum_pv, np.nan), where=cum_vol!=0)
    # Reset VWAP every ~80 periods (approx 1 month of 6h data) to keep it rolling
    # Use rolling window of 80 for practical monthly VWAP
    vwap_80 = pd.Series(vwap).rolling(window=80, min_periods=20).mean().values
    
    # --- Volume Filter: spike above 1.5x median of last 30 periods ---
    vol_median = pd.Series(volume_6h).rolling(window=30, min_processes=10).median().values
    vol_threshold = vol_median * 1.5
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    # Start after warmup period
    start_idx = 80  # for VWAP calculation
    
    for i in range(start_idx, n):
        # Skip if any critical values are NaN
        if (np.isnan(vwap_80[i]) or np.isnan(ema34_1d_aligned[i]) or 
            np.isnan(vol_threshold[i])):
            if position != 0:
                # Check stoploss
                if position == 1 and close_6h[i] <= entry_price - 1.5 * (vwap_80[i] - low_6h[i]):
                    signals[i] = 0.0
                    position = 0
                elif position == -1 and close_6h[i] >= entry_price + 1.5 * (high_6h[i] - vwap_80[i]):
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = 0.25 if position == 1 else -0.25
            continue
        
        # Determine 1d trend
        trend_up = close_6h[i] > ema34_1d_aligned[i]
        trend_down = close_6h[i] < ema34_1d_aligned[i]
        
        # Volume filter: spike above 1.5x median
        vol_ok = volume_6h[i] > vol_threshold[i]
        
        # Distance from VWAP as percentage
        vwap_dist_pct = (close_6h[i] - vwap_80[i]) / vwap_80[i]
        
        if position == 0:
            # Look for mean reversion entries: price extended beyond VWAP, counter to trend
            # Long when price significantly below VWAP in uptrend with volume
            if vwap_dist_pct < -0.015 and trend_up and vol_ok:  # 1.5% below VWAP
                signals[i] = 0.25
                position = 1
                entry_price = close_6h[i]
            # Short when price significantly above VWAP in downtrend with volume
            elif vwap_dist_pct > 0.015 and trend_down and vol_ok:  # 1.5% above VWAP
                signals[i] = -0.25
                position = -1
                entry_price = close_6h[i]
        else:
            # Exit when price returns to VWAP or stoploss
            if position == 1:
                # Stoploss: 1.5x the deviation that triggered entry
                if close_6h[i] <= entry_price - 1.5 * (vwap_80[i] - low_6h[i]):
                    signals[i] = 0.0
                    position = 0
                # Exit: price returns to VWAP (within 0.5%)
                elif abs(close_6h[i] - vwap_80[i]) / vwap_80[i] < 0.005:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = 0.25
            elif position == -1:
                # Stoploss: 1.5x the deviation that triggered entry
                if close_6h[i] >= entry_price + 1.5 * (high_6h[i] - vwap_80[i]):
                    signals[i] = 0.0
                    position = 0
                # Exit: price returns to VWAP (within 0.5%)
                elif abs(close_6h[i] - vwap_80[i]) / vwap_80[i] < 0.005:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = -0.25
    
    return signals
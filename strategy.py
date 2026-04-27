#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h VWAP deviation with 1d trend filter and volume confirmation
# Buy when price > VWAP + 0.5*std in uptrend (price > 1d EMA50), sell when price < VWAP - 0.5*std in downtrend
# Uses 1d EMA50 for trend filter to avoid whipsaws in sideways markets
# Volume > 1.3x 20-period average confirms conviction
# Mean reversion in trends: price tends to revert to VWAP during strong trends
# Target: 20-30 trades/year per symbol (80-120 total over 4 years)

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1d data for trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 40:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    
    # Calculate VWAP and standard deviation for 6h data
    typical_price = (high + low + close) / 3.0
    vwap_numerator = np.cumsum(typical_price * volume)
    vwap_denominator = np.cumsum(volume)
    vwap = np.where(vwap_denominator != 0, vwap_numerator / vwap_denominator, typical_price)
    
    # VWAP deviation (standard deviation of price-VWAP over 20 periods)
    price_vwap_diff = typical_price - vwap
    price_vwap_diff_series = pd.Series(price_vwap_diff)
    vwap_std = price_vwap_diff_series.rolling(window=20, min_periods=20).std().values
    
    # 50-period EMA on daily close for trend filter
    ema50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema50_1d)
    
    # Volume filter: volume > 1.3x 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_filter = volume > (vol_ma * 1.3)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after warmup period
    start_idx = 50
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(vwap[i]) or np.isnan(vwap_std[i]) or 
            np.isnan(ema50_1d_aligned[i]) or np.isnan(vol_ma[i])):
            signals[i] = 0.0
            continue
        
        # Long conditions: price > VWAP + 0.5*vwap_std AND price above daily EMA50 (uptrend) AND volume confirmation
        if (close[i] > vwap[i] + 0.5 * vwap_std[i] and 
            close[i] > ema50_1d_aligned[i] and volume_filter[i]):
            signals[i] = 0.25
            position = 1
        # Short conditions: price < VWAP - 0.5*vwap_std AND price below daily EMA50 (downtrend) AND volume confirmation
        elif (close[i] < vwap[i] - 0.5 * vwap_std[i] and 
              close[i] < ema50_1d_aligned[i] and volume_filter[i]):
            signals[i] = -0.25
            position = -1
        else:
            # Hold current position or flat
            if position == 1:
                signals[i] = 0.25
            elif position == -1:
                signals[i] = -0.25
            else:
                signals[i] = 0.0
    
    return signals

name = "6h_VWAPDev_1dTrend_Volume"
timeframe = "6h"
leverage = 1.0
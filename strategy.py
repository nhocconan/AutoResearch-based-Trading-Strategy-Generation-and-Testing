#!/usr/bin/env python3
# 1d_cci_trend_1w_volume_v1
# Hypothesis: On daily timeframe, capture CCI(20) breakouts with volume confirmation aligned with weekly EMA50 trend.
# Works in bull/bear by following weekly trend. Targets 8-20 trades/year via strict CCI threshold (>100/<-100) and volume filter (2.0x average).

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "1d_cci_trend_1w_volume_v1"
timeframe = "1d"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 60:
        return np.zeros(n)
    
    # Price data
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get weekly data for EMA50 trend
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 50:
        return np.zeros(n)
    
    close_1w = df_1w['close'].values
    ema_50_1w = pd.Series(close_1w).ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Align weekly EMA50 to daily
    ema_50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_50_1w)
    
    # Calculate CCI(20) on daily
    tp = (high + low + close) / 3.0
    sma_tp = pd.Series(tp).rolling(window=20, min_periods=20).mean().values
    mad = pd.Series(tp).rolling(window=20, min_periods=20).apply(lambda x: np.mean(np.abs(x - np.mean(x))), raw=True).values
    cci = (tp - sma_tp) / (0.015 * mad)
    
    # Volume filter: daily volume > 2.0x 20-day average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (2.0 * vol_ma)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    # Start after warmup
    start_idx = 50
    
    for i in range(start_idx, n):
        # Skip if data not available
        if (np.isnan(cci[i]) or np.isnan(ema_50_1w_aligned[i]) or 
            np.isnan(volume_spike[i])):
            if position != 0:
                # Hold position until exit conditions met
                pass
            else:
                signals[i] = 0.0
            continue
        
        if position == 1:  # Long position
            # Exit: CCI < 0 OR trend turns bearish (close < weekly EMA50)
            if (cci[i] < 0) or (close[i] < ema_50_1w_aligned[i]):
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit: CCI > 0 OR trend turns bullish (close > weekly EMA50)
            if (cci[i] > 0) or (close[i] > ema_50_1w_aligned[i]):
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat, look for entry
            # Require volume spike
            if volume_spike[i]:
                # Long entry: CCI > 100 AND close > weekly EMA50 (bullish alignment)
                if (cci[i] > 100) and (close[i] > ema_50_1w_aligned[i]):
                    position = 1
                    signals[i] = 0.25
                # Short entry: CCI < -100 AND close < weekly EMA50 (bearish alignment)
                elif (cci[i] < -100) and (close[i] < ema_50_1w_aligned[i]):
                    position = -1
                    signals[i] = -0.25
    
    return signals
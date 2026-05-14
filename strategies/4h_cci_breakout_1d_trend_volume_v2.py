#!/usr/bin/env python3
# 4h_cci_breakout_1d_trend_volume_v2
# Hypothesis: CCI(20) breakout with volume confirmation and 1-day EMA50 trend filter captures strong momentum moves while avoiding counter-trend trades. Reduced position size and stricter entry to lower trade frequency and avoid overtrading. Designed for both bull and bear markets by following the higher timeframe trend.

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "4h_cci_breakout_1d_trend_volume_v2"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Price data
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1d data for EMA50 trend
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    ema_50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Align 1d EMA50 to 4h
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # CCI(20) calculation
    typical_price = (high + low + close) / 3.0
    tp_ma = pd.Series(typical_price).rolling(window=20, min_periods=20).mean().values
    tp_mad = pd.Series(typical_price).rolling(window=20, min_periods=20).apply(lambda x: np.mean(np.abs(x - np.mean(x))), raw=True).values
    cci = (typical_price - tp_ma) / (0.015 * tp_mad)
    
    # Volume filter: 4h volume > 1.5x 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_filter = volume > (1.5 * vol_ma)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    # Start after warmup
    start_idx = 40
    
    for i in range(start_idx, n):
        # Skip if data not available
        if (np.isnan(cci[i]) or np.isnan(ema_50_1d_aligned[i]) or np.isnan(volume_filter[i])):
            if position != 0:
                # Hold position until exit conditions met
                pass
            else:
                signals[i] = 0.0
            continue
        
        if position == 1:  # Long position
            # Exit: CCI < +100 OR price below 1d EMA50
            if (cci[i] < 100) or (close[i] < ema_50_1d_aligned[i]):
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.20  # Reduced position size
                
        elif position == -1:  # Short position
            # Exit: CCI > -100 OR price above 1d EMA50
            if (cci[i] > -100) or (close[i] > ema_50_1d_aligned[i]):
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.20  # Reduced position size
        else:  # Flat, look for entry
            # Long entry: CCI > +100 + volume + price > 1d EMA50
            if (cci[i] > 100) and volume_filter[i] and (close[i] > ema_50_1d_aligned[i]):
                position = 1
                signals[i] = 0.20
            # Short entry: CCI < -100 + volume + price < 1d EMA50
            elif (cci[i] < -100) and volume_filter[i] and (close[i] < ema_50_1d_aligned[i]):
                position = -1
                signals[i] = -0.20
    
    return signals
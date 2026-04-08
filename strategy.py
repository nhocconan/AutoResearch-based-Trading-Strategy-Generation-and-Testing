#!/usr/bin/env python3
"""
6h_1d_cci_reversion_v1
Hypothesis: CCI mean reversion on 6h with 1d trend filter. In both bull and bear markets,
price tends to revert to mean after extreme CCI readings, but only when aligned with
higher timeframe trend to avoid catching falling knives. Uses discrete position sizing
to minimize turnover.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "6h_1d_cci_reversion_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Price data
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    
    # Get 1d data for trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    
    # 1d EMA(50) for trend filter
    close_1d = df_1d['close'].values
    ema_50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    trend_1d_up = close_1d > ema_50_1d
    trend_1d_down = close_1d < ema_50_1d
    
    # Forward fill trend
    trend_1d_up_series = pd.Series(trend_1d_up)
    trend_1d_down_series = pd.Series(trend_1d_down)
    trend_1d_up_ffilled = trend_1d_up_series.ffill().values
    trend_1d_down_ffilled = trend_1d_down_series.ffill().values
    
    # Align 1d trend to 6h
    trend_1d_up_aligned = align_htf_to_ltf(prices, df_1d, trend_1d_up_ffilled)
    trend_1d_down_aligned = align_htf_to_ltf(prices, df_1d, trend_1d_down_ffilled)
    
    # Calculate CCI(20) on 6h
    typical_price = (high + low + close) / 3.0
    tp_ma = pd.Series(typical_price).rolling(window=20, min_periods=20).mean().values
    tp_mad = pd.Series(typical_price).rolling(window=20, min_periods=20).apply(
        lambda x: np.mean(np.abs(x - np.mean(x))), raw=True
    ).values
    # Avoid division by zero
    tp_mad = np.where(tp_mad == 0, 1e-10, tp_mad)
    cci = (typical_price - tp_ma) / (0.015 * tp_mad)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    # Start after warmup
    start_idx = 20
    
    for i in range(start_idx, n):
        # Skip if data not available
        if (np.isnan(trend_1d_up_aligned[i]) or np.isnan(trend_1d_down_aligned[i]) or
            np.isnan(cci[i])):
            if position != 0:
                # Hold position until exit conditions met
                pass
            else:
                signals[i] = 0.0
            continue
        
        if position == 1:  # Long position
            # Exit: CCI crosses above -100 (mean reversion complete) OR trend turns down
            if cci[i] > -100 or trend_1d_down_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25  # Position size
                
        elif position == -1:  # Short position
            # Exit: CCI crosses below +100 (mean reversion complete) OR trend turns up
            if cci[i] < 100 or trend_1d_up_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25  # Position size
        else:  # Flat, look for entry
            # Long entry: CCI < -200 (oversold) + 1d uptrend
            if cci[i] < -200 and trend_1d_up_aligned[i]:
                position = 1
                signals[i] = 0.25
            # Short entry: CCI > +200 (overbought) + 1d downtrend
            elif cci[i] > 200 and trend_1d_down_aligned[i]:
                position = -1
                signals[i] = -0.25
    
    return signals
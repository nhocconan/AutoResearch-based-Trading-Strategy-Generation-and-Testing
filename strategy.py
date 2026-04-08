#!/usr/bin/env python3
"""
6h_12h_1d_cci_trend_follow_v1
Hypothesis: CCI-based trend following on 6h with 12h trend filter and 1d volume confirmation.
- Entry: 6h CCI(20) crosses above +100 in bullish trend OR below -100 in bearish trend
- Trend filter: 12h close > EMA50 for bullish, < EMA50 for bearish
- Volume filter: 6h volume > 1.3x 20-period average to confirm momentum
- Exit: CCI crosses back through zero or trend reverses
- Position sizing: 0.25 long, -0.25 short
- Designed to work in both bull (trend continuation) and bear (mean reversion via CCI extremes)
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "6h_12h_1d_cci_trend_follow_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Price data
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 12h data for trend filter
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 50:
        return np.zeros(n)
    
    # 12h EMA(50) for trend
    close_12h = df_12h['close'].values
    ema_50_12h = pd.Series(close_12h).ewm(span=50, adjust=False, min_periods=50).mean().values
    trend_12h_up = close_12h > ema_50_12h
    trend_12h_down = close_12h < ema_50_12h
    
    # Forward fill trend
    trend_12h_up_series = pd.Series(trend_12h_up)
    trend_12h_down_series = pd.Series(trend_12h_down)
    trend_12h_up_ffilled = trend_12h_up_series.ffill().values
    trend_12h_down_ffilled = trend_12h_down_series.ffill().values
    
    # Align 12h trend to 6h
    trend_12h_up_aligned = align_htf_to_ltf(prices, df_12h, trend_12h_up_ffilled)
    trend_12h_down_aligned = align_htf_to_ltf(prices, df_12h, trend_12h_down_ffilled)
    
    # 6h CCI calculation
    typical_price = (high + low + close) / 3
    tp_series = pd.Series(typical_price)
    ma = tp_series.rolling(window=20, min_periods=20).mean()
    mad = tp_series.rolling(window=20, min_periods=20).apply(lambda x: np.mean(np.abs(x - np.mean(x))), raw=True)
    cci = (tp_series - ma) / (0.015 * mad)
    cci_values = cci.values
    
    # Volume filter
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_filter = volume > (1.3 * vol_ma)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    # Start after warmup
    start_idx = 50
    
    for i in range(start_idx, n):
        # Skip if data not available
        if (np.isnan(trend_12h_up_aligned[i]) or np.isnan(trend_12h_down_aligned[i]) or
            np.isnan(cci_values[i]) or np.isnan(volume_filter[i])):
            if position != 0:
                # Hold position until exit conditions met
                pass
            else:
                signals[i] = 0.0
            continue
        
        if position == 1:  # Long position
            # Exit: CCI crosses below 0 OR 12h trend turns down
            if (cci_values[i] < 0) or trend_12h_down_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25  # Position size
                
        elif position == -1:  # Short position
            # Exit: CCI crosses above 0 OR 12h trend turns up
            if (cci_values[i] > 0) or trend_12h_up_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25  # Position size
        else:  # Flat, look for entry
            # Long entry: CCI crosses above +100 + 12h uptrend + volume
            if (cci_values[i] > 100) and trend_12h_up_aligned[i] and volume_filter[i]:
                # Confirm crossover from below
                if i > start_idx and cci_values[i-1] <= 100:
                    position = 1
                    signals[i] = 0.25
            # Short entry: CCI crosses below -100 + 12h downtrend + volume
            elif (cci_values[i] < -100) and trend_12h_down_aligned[i] and volume_filter[i]:
                # Confirm crossover from above
                if i > start_idx and cci_values[i-1] >= -100:
                    position = -1
                    signals[i] = -0.25
    
    return signals
#!/usr/bin/env python3
# 4h_cci_breakout_1d_trend_volume_v1
# Hypothesis: On 4h timeframe, CCI(20) crossing above/below +100/-100 with volume expansion and 1d EMA50 trend alignment captures momentum moves. Trend filter avoids counter-trend entries in ranging markets. Volume confirmation filters false breakouts. Designed for both bull and bear markets.
# Entry: Long when CCI > +100 + volume > 1.5x 20-period average + price > 1d EMA50
# Entry: Short when CCI < -100 + volume > 1.5x 20-period average + price < 1d EMA50
# Exit: CCI crosses back below +100 (long) or above -100 (short) or trend reversal
# Position sizing: 0.25 long, -0.25 short

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "4h_cci_breakout_1d_trend_volume_v1"
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
                signals[i] = 0.25  # Position size
                
        elif position == -1:  # Short position
            # Exit: CCI > -100 OR price above 1d EMA50
            if (cci[i] > -100) or (close[i] > ema_50_1d_aligned[i]):
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25  # Position size
        else:  # Flat, look for entry
            # Long entry: CCI > +100 + volume + price > 1d EMA50
            if (cci[i] > 100) and volume_filter[i] and (close[i] > ema_50_1d_aligned[i]):
                position = 1
                signals[i] = 0.25
            # Short entry: CCI < -100 + volume + price < 1d EMA50
            elif (cci[i] < -100) and volume_filter[i] and (close[i] < ema_50_1d_aligned[i]):
                position = -1
                signals[i] = -0.25
    
    return signals
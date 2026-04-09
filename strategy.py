#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Camarilla pivot breakout with 1d trend filter and volume confirmation
# - Uses 1d HTF for trend direction (close > EMA50 = uptrend, close < EMA50 = downtrend)
# - 4h Camarilla pivot levels (R3, S3, R4, S4) from prior 4h bar
# - Long on break above R4 in uptrend, short on break below S4 in downtrend
# - Volume confirmation: current 4h volume > 1.5x 20-period average
# - Fixed position size 0.25 to control drawdown
# - Target: 20-50 trades/year on 4h timeframe (80-200 total over 4 years)
# - Works in bull/bear: trend filter adapts, volume confirms institutional interest

name = "4h_1d_camarilla_breakout_trend_v1"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Load 1d data ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    
    # Calculate 1d EMA50 for trend filter
    ema_50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Align 1d EMA50 to 4h timeframe (wait for completed 1d bar)
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # Pre-compute 4h Camarilla pivot levels from prior bar
    typical_4h = (high + low + close) / 3.0
    range_4h = high - low
    
    # Camarilla levels (based on prior bar)
    r3_4h = typical_4h + range_4h * 1.1 / 4
    s3_4h = typical_4h - range_4h * 1.1 / 4
    r4_4h = typical_4h + range_4h * 1.1 / 2
    s4_4h = typical_4h - range_4h * 1.1 / 2
    
    # Pre-compute volume confirmation (20-period average for 4h)
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(100, n):
        # Skip if any required data is invalid
        if (np.isnan(ema_50_1d_aligned[i]) or np.isnan(r4_4h[i]) or
            np.isnan(s4_4h[i]) or np.isnan(vol_ma_20[i]) or
            vol_ma_20[i] <= 0):
            signals[i] = 0.0
            continue
        
        # Volume confirmation: current 4h volume > 1.5x average
        volume_confirmed = volume[i] > 1.5 * vol_ma_20[i]
        
        # Trend filter: 1d close > EMA50 = uptrend, < = downtrend
        uptrend = close_1d[i] > ema_50_1d[i]  # Use current 1d bar close
        downtrend = close_1d[i] < ema_50_1d[i]  # Use current 1d bar close
        
        # Fixed position size
        position_size = 0.25
        
        if position == 1:  # Long position
            # Exit when price closes below 1d EMA50 (trend change)
            if close[i] < ema_50_1d_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = position_size
                
        elif position == -1:  # Short position
            # Exit when price closes above 1d EMA50 (trend change)
            if close[i] > ema_50_1d_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -position_size
        else:  # Flat
            # Breakout entry with volume confirmation and trend alignment
            if volume_confirmed:
                # Long: break above R4 in uptrend
                if uptrend and close[i] > r4_4h[i]:
                    position = 1
                    signals[i] = position_size
                # Short: break below S4 in downtrend
                elif downtrend and close[i] < s4_4h[i]:
                    position = -1
                    signals[i] = -position_size
    
    return signals
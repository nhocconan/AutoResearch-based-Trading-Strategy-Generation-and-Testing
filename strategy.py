#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1d Camarilla pivot breakout with 1w trend filter and volume confirmation
# Camarilla pivot levels (R1, S1) from prior day identify intraday support/resistance
# Long when price breaks above R1 with volume spike and 1w EMA(34) uptrend
# Short when price breaks below S1 with volume spike and 1w EMA(34) downtrend
# Uses discrete position sizing (0.25) to minimize fee churn
# Targets 20-30 trades/year (80-120 total over 4 years) to stay within fee drag limits for 1d timeframe

name = "1d_Camarilla_R1S1_Breakout_1wEMA34_Trend_VolumeSpike_v1"
timeframe = "1d"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Load 1w data ONCE before loop for EMA trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 50:
        return np.zeros(n)
    
    # Calculate 1w EMA(34) for trend filter
    ema_1w = pd.Series(df_1w['close'].values).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_1w)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after warmup (need at least 2 days for Camarilla calculation)
    start_idx = 2
    
    for i in range(start_idx, n):
        # Calculate Camarilla pivot levels from previous day (i-1)
        # Only use data up to i-1 to avoid look-ahead
        if i-1 < 0:
            signals[i] = 0.0
            continue
            
        high_prev = high[i-1]
        low_prev = low[i-1]
        close_prev = close[i-1]
        
        # Camarilla levels
        pivot = (high_prev + low_prev + close_prev) / 3
        range_prev = high_prev - low_prev
        R1 = close_prev + (range_prev * 1.1 / 12)
        S1 = close_prev - (range_prev * 1.1 / 12)
        
        # Check for NaN values in trend filter
        if np.isnan(ema_1w_aligned[i]):
            signals[i] = 0.0
            continue
        
        # Volume spike (2.0x 20-period average, shifted to avoid look-ahead)
        if i >= 20:
            vol_ma = np.mean(volume[i-20:i])
            volume_spike = volume[i] > (vol_ma * 2.0)
        else:
            volume_spike = False
        
        if position == 0:  # Flat - look for new entries
            # Long: price breaks above R1 + volume spike + 1w EMA uptrend
            if close[i] > R1 and volume_spike and close[i] > ema_1w_aligned[i]:
                signals[i] = 0.25
                position = 1
            # Short: price breaks below S1 + volume spike + 1w EMA downtrend
            elif close[i] < S1 and volume_spike and close[i] < ema_1w_aligned[i]:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        
        elif position == 1:  # Long position
            # Exit: price breaks below S1 or 1w EMA trend turns down
            if close[i] < S1 or close[i] < ema_1w_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:  # Short position
            # Exit: price breaks above R1 or 1w EMA trend turns up
            if close[i] > R1 or close[i] > ema_1w_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals
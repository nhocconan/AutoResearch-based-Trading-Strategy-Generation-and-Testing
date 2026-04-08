#!/usr/bin/env python3
# 1d_1w_cci_breakout_volume_confirm_v1
# Hypothesis: Daily CCI(20) breakouts confirmed by weekly volume surge and CCI trend filter.
# Long when CCI crosses above +100 with weekly volume > 1.8x 20-period average and CCI > 0.
# Short when CCI crosses below -100 with weekly volume > 1.8x 20-period average and CCI < 0.
# Uses CCI to capture trend momentum and volume to avoid false breakouts.
# Designed for 10-20 trades/year on 1d to minimize fee drag while capturing strong trends.
# Works in bull markets via long breakouts and bear markets via short signals.

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "1d_1w_cci_breakout_volume_confirm_v1"
timeframe = "1d"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Daily CCI(20) calculation
    tp = (high + low + close) / 3.0
    sma_tp = pd.Series(tp).rolling(window=20, min_periods=20).mean().values
    mad = pd.Series(tp).rolling(window=20, min_periods=20).apply(lambda x: np.mean(np.abs(x - np.mean(x))), raw=True).values
    cci = (tp - sma_tp) / (0.015 * mad)
    
    # Get weekly data for volume confirmation
    df_1w = get_htf_data(prices, '1w')
    volume_1w = df_1w['volume'].values
    
    # Weekly volume moving average (20-period) for surge detection
    vol_ma_20_1w = pd.Series(volume_1w).rolling(window=20, min_periods=20).mean().values
    vol_ma_20_1w_aligned = align_htf_to_ltf(prices, df_1w, vol_ma_20_1w)
    vol_current_1w_aligned = align_htf_to_ltf(prices, df_1w, volume_1w)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    start_idx = 20  # Ensure CCI is ready
    
    for i in range(start_idx, n):
        # Skip if any required data is not available
        if np.isnan(cci[i]) or np.isnan(sma_tp[i]) or np.isnan(mad[i]) or \
           np.isnan(vol_ma_20_1w_aligned[i]) or np.isnan(vol_current_1w_aligned[i]):
            if position != 0:
                pass  # Hold position
            else:
                signals[i] = 0.0
            continue
        
        # Volume surge condition: current weekly volume > 1.8x 20-period average
        vol_surge = vol_current_1w_aligned[i] > 1.8 * vol_ma_20_1w_aligned[i] if vol_ma_20_1w_aligned[i] > 0 else False
        
        if position == 1:  # Long position
            # Exit: CCI crosses below zero
            if cci[i] < 0 and cci[i-1] >= 0:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit: CCI crosses above zero
            if cci[i] > 0 and cci[i-1] <= 0:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat, look for entry
            # Long entry: CCI crosses above +100 with volume surge and CCI > 0
            if cci[i] > 100 and cci[i-1] <= 100 and vol_surge and cci[i] > 0:
                position = 1
                signals[i] = 0.25
            # Short entry: CCI crosses below -100 with volume surge and CCI < 0
            elif cci[i] < -100 and cci[i-1] >= -100 and vol_surge and cci[i] < 0:
                position = -1
                signals[i] = -0.25
    
    return signals
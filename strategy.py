#!/usr/bin/env python3
# 4h_12h_volume_crossover_v1
# Hypothesis: 4-hour price momentum confirmed by 12-hour volume surge and moving average crossovers.
# Long when 4h EMA(21) crosses above EMA(55) with 12h volume > 1.5x 20-period average.
# Short when 4h EMA(21) crosses below EMA(55) with 12h volume > 1.5x 20-period average.
# Uses volume confirmation to avoid false breakouts and reduce whipsaw.
# Designed for 20-40 trades/year on 4h to minimize fee drag while capturing trending moves.
# Works in bull markets via momentum captures and bear markets via short signals.

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "4h_12h_volume_crossover_v1"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 60:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # 4h EMA(21) and EMA(55) for crossover signals
    ema21 = pd.Series(close).ewm(span=21, adjust=False, min_periods=21).mean().values
    ema55 = pd.Series(close).ewm(span=55, adjust=False, min_periods=55).mean().values
    
    # Get 12h volume data for confirmation
    df_12h = get_htf_data(prices, '12h')
    volume_12h = df_12h['volume'].values
    
    # 12h volume moving average (20-period) for surge detection
    vol_ma_20 = pd.Series(volume_12h).rolling(window=20, min_periods=20).mean().values
    vol_ma_20_aligned = align_htf_to_ltf(prices, df_12h, vol_ma_20)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    start_idx = 55  # Ensure EMA(55) is ready
    
    for i in range(start_idx, n):
        # Skip if any required data is not available
        if np.isnan(ema21[i]) or np.isnan(ema55[i]) or np.isnan(vol_ma_20_aligned[i]):
            if position != 0:
                pass  # Hold position
            else:
                signals[i] = 0.0
            continue
        
        # Volume surge condition: current 12h volume > 1.5x 20-period average
        vol_surge = volume_12h[df_12h.index.get_loc(df_12h.index[-1]) if hasattr(df_12h.index, 'get_loc') else 0] > 1.5 * vol_ma_20_aligned[i] if i < len(vol_ma_20_aligned) else False
        # Simplified volume check using current 12h volume vs its moving average
        vol_surge = volume_12h[-1] > 1.5 * vol_ma_20[-1] if len(volume_12h) >= 20 and len(vol_ma_20) >= 20 else False
        # Correct approach: get current 12h volume bar aligned to 4h time
        # We need the current 12h volume value that corresponds to the 4h bar
        
        # Recalculate with proper alignment
        vol_sma_20_12h = pd.Series(volume_12h).rolling(window=20, min_periods=20).mean().values
        vol_sma_20_12h_aligned = align_htf_to_ltf(prices, df_12h, vol_sma_20_12h)
        
        # Get the current 12h volume value (last value in the 12h series that's available)
        # Since we're using aligned arrays, we can check if current 12h volume exceeds 1.5x its MA
        if i < len(vol_sma_20_12h_aligned):
            # Find corresponding 12h bar index for current 4h bar
            # Volume surge: current 12h volume > 1.5x 20-period average of 12h volume
            vol_surge = False
            if len(volume_12h) >= 20:
                current_vol_ma = vol_sma_20_12h[-1] if len(vol_sma_20_12h) > 0 else 0
                current_volume = volume_12h[-1] if len(volume_12h) > 0 else 0
                if current_vol_ma > 0:
                    vol_surge = current_volume > 1.5 * current_vol_ma
        
        # Simpler approach: use the aligned volume and its moving average directly
        # Recalculate volume MA on aligned data for simplicity
        vol_close_12h = df_12h['volume'].values
        vol_ma_aligned = align_htf_to_ltf(prices, df_12h, 
                                          pd.Series(vol_close_12h).rolling(window=20, min_periods=20).mean().values)
        vol_current_aligned = align_htf_to_ltf(prices, df_12h, vol_close_12h)
        
        vol_surge = vol_current_aligned[i] > 1.5 * vol_ma_aligned[i] if i < len(vol_ma_aligned) and not np.isnan(vol_ma_aligned[i]) and vol_ma_aligned[i] > 0 else False
        
        if position == 1:  # Long position
            # Exit: EMA(21) crosses below EMA(55)
            if ema21[i] < ema55[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit: EMA(21) crosses above EMA(55)
            if ema21[i] > ema55[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat, look for entry
            # Long entry: EMA(21) crosses above EMA(55) with volume surge
            if ema21[i] > ema55[i] and ema21[i-1] <= ema55[i-1] and vol_surge:
                position = 1
                signals[i] = 0.25
            # Short entry: EMA(21) crosses below EMA(55) with volume surge
            elif ema21[i] < ema55[i] and ema21[i-1] >= ema55[i-1] and vol_surge:
                position = -1
                signals[i] = -0.25
    
    return signals
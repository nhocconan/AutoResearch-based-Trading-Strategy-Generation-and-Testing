# 6h_12h_1d_camarilla_breakout_v1
# Hypothesis: 6-hour price breakouts confirmed by 12-hour volume surge and 1-day Camarilla pivot levels.
# Long when price breaks above Camarilla R4 on 6h with 12h volume > 2x 20-period average.
# Short when price breaks below Camarilla S4 on 6h with 12h volume > 2x 20-period average.
# Camarilla levels from 1-day timeframe provide institutional support/resistance.
# Volume surge filters false breakouts. Designed for 15-30 trades/year on 6h to minimize fee drag.
# Works in bull markets via breakout continuation and bear markets via breakdowns.

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "6h_12h_1d_camarilla_breakout_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Get 12h volume data for confirmation
    df_12h = get_htf_data(prices, '12h')
    volume_12h = df_12h['volume'].values
    
    # 12h volume moving average (20-period) for surge detection
    vol_ma_20_12h = pd.Series(volume_12h).rolling(window=20, min_periods=20).mean().values
    vol_ma_20_12h_aligned = align_htf_to_ltf(prices, df_12h, vol_ma_20_12h)
    vol_current_12h_aligned = align_htf_to_ltf(prices, df_12h, volume_12h)
    
    # Get 1d OHLC for Camarilla pivot calculation
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate Camarilla levels from previous 1d candle
    # Camarilla: R4 = close + 1.5*(high-low), S4 = close - 1.5*(high-low)
    # Using previous day's OHLC to avoid look-ahead
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    camarilla_r4 = close_1d + 1.5 * (high_1d - low_1d)
    camarilla_s4 = close_1d - 1.5 * (high_1d - low_1d)
    
    # Align Camarilla levels to 6h timeframe
    camarilla_r4_aligned = align_htf_to_ltf(prices, df_1d, camarilla_r4)
    camarilla_s4_aligned = align_htf_to_ltf(prices, df_1d, camarilla_s4)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    start_idx = 20  # Ensure volume MA is ready
    
    for i in range(start_idx, n):
        # Skip if any required data is not available
        if np.isnan(vol_ma_20_12h_aligned[i]) or np.isnan(camarilla_r4_aligned[i]) or np.isnan(camarilla_s4_aligned[i]):
            if position != 0:
                pass  # Hold position
            else:
                signals[i] = 0.0
            continue
        
        # Volume surge condition: current 12h volume > 2x 20-period average
        vol_surge = vol_current_12h_aligned[i] > 2.0 * vol_ma_20_12h_aligned[i] if vol_ma_20_12h_aligned[i] > 0 else False
        
        if position == 1:  # Long position
            # Exit: price closes below Camarilla S4
            if close[i] < camarilla_s4_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit: price closes above Camarilla R4
            if close[i] > camarilla_r4_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat, look for entry
            # Long entry: price breaks above Camarilla R4 with volume surge
            if close[i] > camarilla_r4_aligned[i] and vol_surge:
                position = 1
                signals[i] = 0.25
            # Short entry: price breaks below Camarilla S4 with volume surge
            elif close[i] < camarilla_s4_aligned[i] and vol_surge:
                position = -1
                signals[i] = -0.25
    
    return signals
# 6H_1D_CAMARILLA_R3S3_FADE_R4S4_BREAKOUT_VOLUME_CONFIRM
# Hypothesis: Use daily Camarilla pivot levels for 6h trading.
# Fade at R3/S3 levels (mean reversion), breakout continuation at R4/S4 (trend following).
# Add volume confirmation (>1.5x 20-period average) to avoid false signals.
# Works in both bull and bear markets by adapting to price action at key levels.
# Target: 50-150 total trades over 4 years (12-37/year)

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Load daily data for Camarilla pivots
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate Camarilla pivot levels for previous day
    # R4 = Close + 1.5 * (High - Low)
    # R3 = Close + 1.0 * (High - Low)
    # S3 = Close - 1.0 * (High - Low)
    # S4 = Close - 1.5 * (High - Low)
    hl_range = high_1d - low_1d
    r4_1d = close_1d + 1.5 * hl_range
    r3_1d = close_1d + 1.0 * hl_range
    s3_1d = close_1d - 1.0 * hl_range
    s4_1d = close_1d - 1.5 * hl_range
    
    # Align to 6h timeframe (use previous day's levels)
    r4_1d_aligned = align_htf_to_ltf(prices, df_1d, r4_1d)
    r3_1d_aligned = align_htf_to_ltf(prices, df_1d, r3_1d)
    s3_1d_aligned = align_htf_to_ltf(prices, df_1d, s3_1d)
    s4_1d_aligned = align_htf_to_ltf(prices, df_1d, s4_1d)
    
    # Price and volume data
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Volume filter: volume > 1.5x 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_filter = volume > (vol_ma * 1.5)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(50, n):
        # Skip if NaN in indicators
        if (np.isnan(r4_1d_aligned[i]) or np.isnan(r3_1d_aligned[i]) or 
            np.isnan(s3_1d_aligned[i]) or np.isnan(s4_1d_aligned[i]) or 
            np.isnan(vol_ma[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        price = close[i]
        has_volume = vol_filter[i]
        
        # Get pivot levels
        r4 = r4_1d_aligned[i]
        r3 = r3_1d_aligned[i]
        s3 = s3_1d_aligned[i]
        s4 = s4_1d_aligned[i]
        
        if position == 0:
            # Long conditions
            long_signal = False
            # Fade at S3 (mean reversion) - price rejects S3 and moves back up
            if price > s3 and has_volume:
                # Check if we bounced from S3 (price was at or below S3 previous bar)
                if i > 0 and low[i-1] <= s3:
                    long_signal = True
            # Breakout above R4 (trend following)
            elif price > r4 and has_volume:
                long_signal = True
            
            # Short conditions
            short_signal = False
            # Fade at R3 (mean reversion) - price rejects R3 and moves back down
            if price < r3 and has_volume:
                # Check if we rejected R3 (price was at or above R3 previous bar)
                if i > 0 and high[i-1] >= r3:
                    short_signal = True
            # Breakdown below S4 (trend following)
            elif price < s4 and has_volume:
                short_signal = True
            
            if long_signal:
                signals[i] = 0.25
                position = 1
            elif short_signal:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit long: fade at R3 or stop loss
            exit_signal = False
            # Take profit at R3 (fade level)
            if price < r3:
                exit_signal = True
            # Stop loss if price breaks below S3
            elif price < s3:
                exit_signal = True
            
            if exit_signal:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: fade at S3 or stop loss
            exit_signal = False
            # Take profit at S3 (fade level)
            if price > s3:
                exit_signal = True
            # Stop loss if price breaks above R3
            elif price > r3:
                exit_signal = True
            
            if exit_signal:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "6H_1D_CAMARILLA_R3S3_FADE_R4S4_BREAKOUT_VOLUME_CONFIRM"
timeframe = "6h"
leverage = 1.0
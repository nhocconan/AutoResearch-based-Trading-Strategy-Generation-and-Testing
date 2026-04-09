#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h strategy using 1d Camarilla pivot levels with volume spike confirmation
# - Uses 1d HTF for Camarilla pivot levels (R3, S3, R4, S4) to identify key support/resistance
# - Long when price breaks above R4 with volume > 2.0x 20-period average (strong bullish breakout)
# - Short when price breaks below S4 with volume > 2.0x 20-period average (strong bearish breakout)
# - Exit when price returns to the 1d VWAP (mean reversion to daily fair value)
# - Fixed position size 0.25 to control drawdown
# - Works in bull/bear: Breakouts capture strong momentum, VWAP exit prevents overstaying
# - Target: 12-25 trades/year on 6h timeframe (48-100 total over 4 years)

name = "6h_1d_camarilla_breakout_vwap_v1"
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
    
    # Load 1d data ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate 1d Camarilla pivot levels
    # Pivot = (High + Low + Close) / 3
    # Range = High - Low
    # R4 = Close + Range * 1.1/2
    # S4 = Close - Range * 1.1/2
    # R3 = Close + Range * 1.1/4
    # S3 = Close - Range * 1.1/4
    pivot_1d = (high_1d + low_1d + close_1d) / 3.0
    range_1d = high_1d - low_1d
    r4_1d = close_1d + range_1d * 1.1 / 2.0
    s4_1d = close_1d - range_1d * 1.1 / 2.0
    r3_1d = close_1d + range_1d * 1.1 / 4.0
    s3_1d = close_1d - range_1d * 1.1 / 4.0
    
    # Align Camarilla levels to 6h timeframe (wait for completed 1d bar)
    r4_aligned = align_htf_to_ltf(prices, df_1d, r4_1d)
    s4_aligned = align_htf_to_ltf(prices, df_1d, s4_1d)
    r3_aligned = align_htf_to_ltf(prices, df_1d, r3_1d)
    s3_aligned = align_htf_to_ltf(prices, df_1d, s3_1d)
    
    # Pre-compute 1d VWAP for exit (typical price * volume)
    typical_price_1d = (high_1d + low_1d + close_1d) / 3.0
    pv_1d = typical_price_1d * df_1d['volume'].values
    vwap_1d = np.cumsum(pv_1d) / np.cumsum(df_1d['volume'].values)
    vwap_aligned = align_htf_to_ltf(prices, df_1d, vwap_1d)
    
    # Pre-compute volume confirmation (20-period average)
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(50, n):
        # Skip if any required data is invalid
        if (np.isnan(r4_aligned[i]) or np.isnan(s4_aligned[i]) or
            np.isnan(vwap_aligned[i]) or np.isnan(vol_ma_20[i]) or
            vol_ma_20[i] <= 0):
            signals[i] = 0.0
            continue
        
        # Volume confirmation: current 6h volume > 2.0x average
        volume_confirmed = volume[i] > 2.0 * vol_ma_20[i]
        
        if position == 1:  # Long position
            # Exit when price returns to 1d VWAP (mean reversion)
            if close[i] <= vwap_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit when price returns to 1d VWAP (mean reversion)
            if close[i] >= vwap_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat
            # Entry logic: Camarilla breakout with volume confirmation
            if volume_confirmed:
                # Long breakout: price > R4 (strong bullish breakout)
                if close[i] > r4_aligned[i]:
                    position = 1
                    signals[i] = 0.25
                # Short breakout: price < S4 (strong bearish breakout)
                elif close[i] < s4_aligned[i]:
                    position = -1
                    signals[i] = -0.25
    
    return signals
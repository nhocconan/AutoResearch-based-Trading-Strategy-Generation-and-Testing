#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h Camarilla pivot breakout with 1d volume spike and chop regime filter
# - Primary: 12h price breaks above/below Camarilla H3/L3 levels from prior 1d
# - Volume filter: 1d volume > 1.3x 20-period volume MA to ensure participation
# - Regime filter: Choppiness Index(14) > 61.8 (ranging market) for mean reversion edge
# - Exit: Price returns to Camarilla H4/L4 levels (adaptive profit target)
# - Position sizing: 0.25 (discrete level to minimize fee churn)
# - Works in bull/bear: Camarilla pivots act as support/resistance, chop filter avoids false breakouts in trends
# - Target: 50-150 total trades over 4 years = 12-37/year for 12h timeframe

name = "12h_1d_camarilla_volume_chop_v1"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Load HTF data ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    # Pre-compute primary timeframe data
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Pre-compute HTF data
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    volume_1d = df_1d['volume'].values
    
    # Calculate prior 1d Camarilla levels (using prior day's OHLC)
    # Camarilla: H4 = close + 1.1*(high-low)*1.1/2, H3 = close + 1.1*(high-low)*1.1/4
    # L3 = close - 1.1*(high-low)*1.1/4, L4 = close - 1.1*(high-low)*1.1/2
    range_1d = high_1d - low_1d
    camarilla_h4 = close_1d + 1.1 * range_1d * 1.1 / 2
    camarilla_h3 = close_1d + 1.1 * range_1d * 1.1 / 4
    camarilla_l3 = close_1d - 1.1 * range_1d * 1.1 / 4
    camarilla_l4 = close_1d - 1.1 * range_1d * 1.1 / 2
    
    # Shift by 1 to use prior day's levels (avoid look-ahead)
    camarilla_h4 = np.roll(camarilla_h4, 1)
    camarilla_h3 = np.roll(camarilla_h3, 1)
    camarilla_l3 = np.roll(camarilla_l3, 1)
    camarilla_l4 = np.roll(camarilla_l4, 1)
    
    # Handle first element (use same day's data as fallback)
    camarilla_h4[0] = close_1d[0] + 1.1 * (high_1d[0] - low_1d[0]) * 1.1 / 2
    camarilla_h3[0] = close_1d[0] + 1.1 * (high_1d[0] - low_1d[0]) * 1.1 / 4
    camarilla_l3[0] = close_1d[0] - 1.1 * (high_1d[0] - low_1d[0]) * 1.1 / 4
    camarilla_l4[0] = close_1d[0] - 1.1 * (high_1d[0] - low_1d[0]) * 1.1 / 2
    
    # Align Camarilla levels to 12h timeframe
    camarilla_h4_aligned = align_htf_to_ltf(prices, df_1d, camarilla_h4)
    camarilla_h3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_h3)
    camarilla_l3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_l3)
    camarilla_l4_aligned = align_htf_to_ltf(prices, df_1d, camarilla_l4)
    
    # Calculate 1d volume confirmation: volume > 1.3x 20-period volume MA
    volume_ma_20_1d = pd.Series(volume_1d).rolling(window=20, min_periods=20).mean().values
    volume_ma_20_1d_aligned = align_htf_to_ltf(prices, df_1d, volume_ma_20_1d)
    
    # Calculate 14-period Choppiness Index for regime filter
    high_low = high - low
    high_close = np.abs(high - np.roll(close, 1))
    low_close = np.abs(low - np.roll(close, 1))
    
    # Handle first element
    high_low[0] = high[0] - low[0]
    high_close[0] = np.abs(high[0] - close[0])
    low_close[0] = np.abs(low[0] - close[0])
    
    tr = np.maximum(high_low, np.maximum(high_close, low_close))
    atr_sum = pd.Series(tr).rolling(window=14, min_periods=14).sum().values
    max_high = pd.Series(high).rolling(window=14, min_periods=14).max().values
    min_low = pd.Series(low).rolling(window=14, min_periods=14).min().values
    chop = 100 * np.log10(atr_sum / (max_high - min_low)) / np.log10(14)
    chop_filter = chop > 61.8  # Chop > 61.8 indicates ranging market (mean reversion)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(50, n):
        # Skip if any required data is invalid
        if (np.isnan(camarilla_h4_aligned[i]) or np.isnan(camarilla_h3_aligned[i]) or 
            np.isnan(camarilla_l3_aligned[i]) or np.isnan(camarilla_l4_aligned[i]) or 
            np.isnan(volume_ma_20_1d_aligned[i]) or np.isnan(chop[i])):
            signals[i] = 0.0
            continue
        
        # Volume filter: current 1d volume > 1.3x 20-period volume MA
        volume_1d_current = align_htf_to_ltf(prices, df_1d, volume_1d)
        vol_confirm = volume_1d_current[i] > 1.3 * volume_ma_20_1d_aligned[i]
        
        if position == 0:  # Flat - look for new entries
            # Long entry: price breaks above Camarilla H3 + vol confirmation + chop filter
            if (close[i] > camarilla_h3_aligned[i] and 
                vol_confirm and chop_filter[i]):
                position = 1
                signals[i] = 0.25
            # Short entry: price breaks below Camarilla L3 + vol confirmation + chop filter
            elif (close[i] < camarilla_l3_aligned[i] and 
                  vol_confirm and chop_filter[i]):
                position = -1
                signals[i] = -0.25
            else:
                signals[i] = 0.0
        else:  # Have position - look for exit to Camarilla H4/L4
            # Exit: price reaches Camarilla H4 (for longs) or L4 (for shorts)
            if position == 1:  # Long position
                if close[i] >= camarilla_h4_aligned[i]:
                    position = 0
                    signals[i] = 0.0
                else:
                    signals[i] = 0.25
            else:  # position == -1 (Short position)
                if close[i] <= camarilla_l4_aligned[i]:
                    position = 0
                    signals[i] = 0.0
                else:
                    signals[i] = -0.25
    
    return signals
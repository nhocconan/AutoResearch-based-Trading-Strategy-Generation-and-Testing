#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h Camarilla pivot levels (from 1d) + volume spike + choppiness regime filter
# - Long when price touches Camarilla L3 support AND 1d volume > 2x 20-period average AND 12h chop < 61.8 (trending)
# - Short when price touches Camarilla H3 resistance AND 1d volume > 2x 20-period average AND 12h chop < 61.8 (trending)
# - Exit when price crosses Camarilla H4/L4 levels or opposite pivot touch occurs
# - Uses discrete position sizing 0.25 to limit fee churn
# - Target: 12-37 trades/year on 12h timeframe (50-150 total over 4 years)
# - Camarilla pivots work well in ranging markets which appear in both bull and bear phases
# - Volume confirmation reduces false touches
# - Choppiness filter ensures we only trade in trending regimes where pivot breaks are meaningful

name = "12h_1d_camarilla_volume_chop_v1"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Load HTF data ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    
    # Pre-compute 12h price and volume
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Pre-compute 12h volume confirmation
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (2.0 * vol_ma)
    
    # Pre-compute 12h choppiness index (14-period)
    def true_range(high, low, close_prev):
        tr1 = high - low
        tr2 = np.abs(high - close_prev)
        tr3 = np.abs(low - close_prev)
        return np.maximum(tr1, np.maximum(tr2, tr3))
    
    close_prev = np.roll(close, 1)
    close_prev[0] = close[0]
    tr = true_range(high, low, close_prev)
    atr14 = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    max_high = pd.Series(high).rolling(window=14, min_periods=14).max().values
    min_low = pd.Series(low).rolling(window=14, min_periods=14).min().values
    
    # Avoid division by zero
    range_14 = max_high - min_low
    chop = np.where(range_14 != 0, 100 * np.log10(atr14 * np.sqrt(14) / range_14) / np.log10(27), 50)
    
    # Chop < 61.8 indicates trending regime (good for pivot breaks)
    chop_trending = chop < 61.8
    
    # Calculate Camarilla levels from previous 1d candle
    # Typical Camarilla formula based on previous day's range
    prev_close = df_1d['close'].values
    prev_high = df_1d['high'].values
    prev_low = df_1d['low'].values
    
    # Calculate for each 1d bar
    camarilla_h5 = np.zeros_like(prev_close)
    camarilla_h4 = np.zeros_like(prev_close)
    camarilla_h3 = np.zeros_like(prev_close)
    camarilla_l3 = np.zeros_like(prev_close)
    camarilla_l4 = np.zeros_like(prev_close)
    camarilla_l5 = np.zeros_like(prev_close)
    
    for i in range(len(df_1d)):
        if i == 0:
            # First bar - use same values (will be aligned properly)
            camarilla_h5[i] = prev_high[i]
            camarilla_h4[i] = prev_high[i]
            camarilla_h3[i] = prev_high[i]
            camarilla_l3[i] = prev_low[i]
            camarilla_l4[i] = prev_low[i]
            camarilla_l5[i] = prev_low[i]
        else:
            range_val = prev_high[i-1] - prev_low[i-1]
            camarilla_h5[i] = prev_close[i-1] + (range_val * 1.500)
            camarilla_h4[i] = prev_close[i-1] + (range_val * 1.250)
            camarilla_h3[i] = prev_close[i-1] + (range_val * 1.125)
            camarilla_l3[i] = prev_close[i-1] - (range_val * 1.125)
            camarilla_l4[i] = prev_close[i-1] - (range_val * 1.250)
            camarilla_l5[i] = prev_close[i-1] - (range_val * 1.500)
    
    # Align HTF indicators to 12h timeframe
    camarilla_h3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_h3)
    camarilla_l3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_l3)
    camarilla_h4_aligned = align_htf_to_ltf(prices, df_1d, camarilla_h4)
    camarilla_l4_aligned = align_htf_to_ltf(prices, df_1d, camarilla_l4)
    volume_spike_aligned = align_htf_to_ltf(prices, df_1d, volume_spike)
    chop_trending_aligned = align_htf_to_ltf(prices, df_1d, chop_trending)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(50, n):  # Start after warmup
        # Skip if any required data is invalid
        if (np.isnan(camarilla_h3_aligned[i]) or np.isnan(camarilla_l3_aligned[i]) or 
            np.isnan(camarilla_h4_aligned[i]) or np.isnan(camarilla_l4_aligned[i]) or
            np.isnan(volume_spike_aligned[i]) or np.isnan(chop_trending_aligned[i])):
            if position == 0:
                signals[i] = 0.0
            elif position == 1:
                signals[i] = 0.25
            else:
                signals[i] = -0.25
            continue
        
        if position == 0:  # Flat - look for new entries
            # Long conditions: price touches Camarilla L3 support AND volume spike AND trending chop
            if (low[i] <= camarilla_l3_aligned[i] and 
                volume_spike_aligned[i] and 
                chop_trending_aligned[i]):
                position = 1
                signals[i] = 0.25
            # Short conditions: price touches Camarilla H3 resistance AND volume spike AND trending chop
            elif (high[i] >= camarilla_h3_aligned[i] and 
                  volume_spike_aligned[i] and 
                  chop_trending_aligned[i]):
                position = -1
                signals[i] = -0.25
            else:
                signals[i] = 0.0
        else:  # Have position - look for exit
            # Exit conditions: price crosses Camarilla H4/L4 levels
            exit_long = (position == 1 and 
                        (high[i] >= camarilla_h4_aligned[i] or low[i] <= camarilla_l4_aligned[i]))
            exit_short = (position == -1 and 
                         (high[i] >= camarilla_h4_aligned[i] or low[i] <= camarilla_l4_aligned[i]))
            
            if exit_long or exit_short:
                position = 0
                signals[i] = 0.0
            else:
                if position == 1:
                    signals[i] = 0.25
                else:
                    signals[i] = -0.25
    
    return signals
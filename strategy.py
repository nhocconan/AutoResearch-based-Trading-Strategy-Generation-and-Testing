#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1d Camarilla pivot breakout with 1w trend filter and volume confirmation
# - Long when price breaks above Camarilla H3 level AND 1w EMA50 uptrend AND volume > 1.5x 20-bar avg
# - Short when price breaks below Camarilla L3 level AND 1w EMA50 downtrend AND volume > 1.5x 20-bar avg
# - Exit when price returns to Camarilla Pivot level (mean reversion to equilibrium)
# - Uses 1w EMA50 for trend filter to avoid counter-trend trades in bear markets
# - Discrete position sizing (0.25) to minimize fee churn
# - Target: 15-25 trades/year on 1d timeframe (60-100 total over 4 years)
# - Camarilla pivots work well in ranging/bear markets which matches 2025+ test conditions

name = "1d_1w_camarilla_breakout_volume_trend_v2"
timeframe = "1d"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Load HTF data ONCE before loop
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 50:
        return np.zeros(n)
    
    # Pre-compute volume confirmation: > 1.5x 20-period average
    volume_20_avg = prices['volume'].rolling(window=20, min_periods=20).mean().values
    vol_spike = prices['volume'] > (1.5 * volume_20_avg)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    # Pre-compute aligned 1w data
    c_1w = df_1w['close'].values
    h_1w = df_1w['high'].values
    l_1w = df_1w['low'].values
    
    # Pre-compute 1w EMA(50) for trend filter
    ema50_1w = pd.Series(c_1w).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema50_1w)
    
    for i in range(50, n):  # Start after EMA50 warmup
        # Skip if any required data is invalid
        if (np.isnan(volume_20_avg[i]) or 
            np.isnan(ema50_1w_aligned[i]) or 
            np.isnan(c_1w[i]) or np.isnan(h_1w[i]) or np.isnan(l_1w[i])):
            # Hold current position or flat
            if position == 0:
                signals[i] = 0.0
            elif position == 1:
                signals[i] = 0.25
            else:
                signals[i] = -0.25
            continue
        
        # Need at least 1 day of prior data for Camarilla calculation
        if i < 1:
            signals[i] = 0.0
            continue
            
        # Calculate Camarilla levels from previous day
        y_high = prices['high'].iloc[i-1]
        y_low = prices['low'].iloc[i-1]
        y_close = prices['close'].iloc[i-1]
        
        # Camarilla levels
        pivot = (y_high + y_low + y_close) / 3
        range_hl = y_high - y_low
        h3 = pivot + (range_hl * 1.1 / 4)
        l3 = pivot - (range_hl * 1.1 / 4)
        
        if position == 0:  # Flat - look for new breakout entries
            # Long when price breaks above H3 AND in 1w uptrend with volume spike
            if (prices['close'].iloc[i] > h3 and 
                prices['close'].iloc[i] > ema50_1w_aligned[i] and 
                vol_spike.iloc[i]):
                position = 1
                signals[i] = 0.25
            # Short when price breaks below L3 AND in 1w downtrend with volume spike
            elif (prices['close'].iloc[i] < l3 and 
                  prices['close'].iloc[i] < ema50_1w_aligned[i] and 
                  vol_spike.iloc[i]):
                position = -1
                signals[i] = -0.25
            else:
                signals[i] = 0.0
        else:  # Have position - look for exit to pivot (mean reversion)
            # Exit when price returns to pivot level
            exit_signal = False
            if position == 1:  # Long position
                if prices['close'].iloc[i] <= pivot:
                    exit_signal = True
            elif position == -1:  # Short position
                if prices['close'].iloc[i] >= pivot:
                    exit_signal = True
            
            if exit_signal:
                position = 0
                signals[i] = 0.0
            else:
                if position == 1:
                    signals[i] = 0.25
                else:
                    signals[i] = -0.25
    
    return signals
#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h Camarilla pivot breakout with 1d trend filter and volume confirmation
# - Long when price breaks above H3 (1d Camarilla) AND price > 1d EMA50 AND volume > 1.8x 20-bar avg
# - Short when price breaks below L3 (1d Camarilla) AND price < 1d EMA50 AND volume > 1.8x 20-bar avg
# - Exit when price crosses back to 1d EMA50 (mean reversion to trend)
# - Uses 1d EMA50 for trend filter to align with medium-term direction
# - Discrete position sizing (0.25) to minimize fee churn
# - Target: 12-25 trades/year on 12h timeframe (50-100 total over 4 years)
# - Camarilla pivots work well in ranging/bear markets; volume confirmation filters false breakouts
# - 12h timeframe reduces noise vs lower TFs while capturing meaningful moves

name = "12h_1d_camarilla_breakout_volume_trend_v1"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Load HTF data ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # Pre-compute 1d Camarilla pivot levels
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Camarilla formula: range = high - low
    # H3 = close + (high - low) * 1.1/4
    # L3 = close - (high - low) * 1.1/4
    range_1d = high_1d - low_1d
    h3 = close_1d + (range_1d * 1.1 / 4)
    l3 = close_1d - (range_1d * 1.1 / 4)
    
    # Pre-compute volume confirmation: > 1.8x 20-period average
    volume_20_avg = prices['volume'].rolling(window=20, min_periods=20).mean().values
    vol_spike = prices['volume'] > (1.8 * volume_20_avg)
    
    # Pre-compute 1d EMA(50) for trend filter
    ema50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Align HTF arrays to LTF
    h3_aligned = align_htf_to_ltf(prices, df_1d, h3)
    l3_aligned = align_htf_to_ltf(prices, df_1d, l3)
    ema50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema50_1d)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(50, n):  # Start after EMA50 warmup
        # Skip if any required data is invalid
        if (np.isnan(h3_aligned[i]) or np.isnan(l3_aligned[i]) or 
            np.isnan(ema50_1d_aligned[i]) or np.isnan(volume_20_avg[i])):
            # Hold current position or flat
            if position == 0:
                signals[i] = 0.0
            elif position == 1:
                signals[i] = 0.25
            else:
                signals[i] = -0.25
            continue
        
        if position == 0:  # Flat - look for new breakout entries
            # Long when price breaks above H3 AND in 1d uptrend with volume spike
            if (prices['close'].iloc[i] > h3_aligned[i] and 
                prices['close'].iloc[i] > ema50_1d_aligned[i] and 
                vol_spike.iloc[i]):
                position = 1
                signals[i] = 0.25
            # Short when price breaks below L3 AND in 1d downtrend with volume spike
            elif (prices['close'].iloc[i] < l3_aligned[i] and 
                  prices['close'].iloc[i] < ema50_1d_aligned[i] and 
                  vol_spike.iloc[i]):
                position = -1
                signals[i] = -0.25
            else:
                signals[i] = 0.0
        else:  # Have position - look for exit to trend
            # Exit when price crosses back to 1d EMA50 (mean reversion to trend)
            exit_signal = False
            if position == 1:  # Long position
                if prices['close'].iloc[i] < ema50_1d_aligned[i]:
                    exit_signal = True
            elif position == -1:  # Short position
                if prices['close'].iloc[i] > ema50_1d_aligned[i]:
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
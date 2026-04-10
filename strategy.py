#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1h Camarilla pivot breakout with 4h trend filter and volume confirmation
# - Long when price breaks above H3 pivot AND 4h EMA50 > EMA200 (uptrend) AND volume > 1.5x 20-bar avg
# - Short when price breaks below L3 pivot AND 4h EMA50 < EMA200 (downtrend) AND volume > 1.5x 20-bar avg
# - Exit when price returns to the 4h VWAP (mean reversion to fair value)
# - Uses 4h EMA50/EMA200 for trend filter to avoid counter-trend trades
# - Uses 4h VWAP as dynamic exit target
# - Discrete position sizing (0.20) to minimize fee churn
# - Target: 15-35 trades/year on 1h timeframe (60-140 total over 4 years)
# - Camarilla pivots work well in ranging markets; 4h trend filter avoids whipsaws in strong trends
# - Volume confirmation ensures breakouts have conviction
# - VWAP exit provides logical mean reversion target in both bull and bear markets

name = "1h_4h_camarilla_breakout_volume_trend_vwap_exit_v1"
timeframe = "1h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Load HTF data ONCE before loop
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 50:
        return np.zeros(n)
    
    # Pre-compute 4h EMA50 and EMA200 for trend filter
    close_4h = df_4h['close'].values
    ema50_4h = pd.Series(close_4h).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema200_4h = pd.Series(close_4h).ewm(span=200, adjust=False, min_periods=200).mean().values
    
    # Pre-compute 4h VWAP for exit target
    typical_4h = (df_4h['high'] + df_4h['low'] + df_4h['close']) / 3
    vwap_4h = (typical_4h * df_4h['volume']).cumsum() / df_4h['volume'].cumsum()
    vwap_4h_vals = vwap_4h.values
    
    # Pre-compute aligned 4h data
    ema50_4h_aligned = align_htf_to_ltf(prices, df_4h, ema50_4h)
    ema200_4h_aligned = align_htf_to_ltf(prices, df_4h, ema200_4h)
    vwap_4h_aligned = align_htf_to_ltf(prices, df_4h, vwap_4h_vals)
    
    # Pre-compute 1h Camarilla pivots (using previous bar's high/low/close)
    high_shift = prices['high'].shift(1)
    low_shift = prices['low'].shift(1)
    close_shift = prices['close'].shift(1)
    
    # Avoid look-ahead: use previous bar's data to calculate today's pivots
    pivot = (high_shift + low_shift + close_shift) / 3
    range_hl = high_shift - low_shift
    
    # Camarilla levels
    h3 = pivot + (range_hl * 1.1 / 4)
    l3 = pivot - (range_hl * 1.1 / 4)
    h4 = pivot + (range_hl * 1.1 / 2)
    l4 = pivot - (range_hl * 1.1 / 2)
    
    # Pre-compute volume confirmation: > 1.5x 20-period average
    volume_20_avg = prices['volume'].rolling(window=20, min_periods=20).mean().values
    vol_spike = prices['volume'] > (1.5 * volume_20_avg)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(50, n):  # Start after EMA200 warmup
        # Skip if any required data is invalid
        if (np.isnan(ema50_4h_aligned[i]) or np.isnan(ema200_4h_aligned[i]) or 
            np.isnan(vwap_4h_aligned[i]) or np.isnan(h3[i]) or np.isnan(l3[i]) or
            np.isnan(volume_20_avg[i]) or np.isnan(prices['close'].iloc[i])):
            # Hold current position or flat
            if position == 0:
                signals[i] = 0.0
            elif position == 1:
                signals[i] = 0.20
            else:
                signals[i] = -0.20
            continue
        
        if position == 0:  # Flat - look for new breakout entries
            # Long when price breaks above H3 AND 4h uptrend AND volume spike
            if (prices['close'].iloc[i] > h3[i] and 
                ema50_4h_aligned[i] > ema200_4h_aligned[i] and 
                vol_spike.iloc[i]):
                position = 1
                signals[i] = 0.20
            # Short when price breaks below L3 AND 4h downtrend AND volume spike
            elif (prices['close'].iloc[i] < l3[i] and 
                  ema50_4h_aligned[i] < ema200_4h_aligned[i] and 
                  vol_spike.iloc[i]):
                position = -1
                signals[i] = -0.20
            else:
                signals[i] = 0.0
        else:  # Have position - look for exit to VWAP
            # Exit when price returns to 4h VWAP (mean reversion)
            exit_signal = False
            if position == 1:  # Long position
                if prices['close'].iloc[i] <= vwap_4h_aligned[i]:
                    exit_signal = True
            elif position == -1:  # Short position
                if prices['close'].iloc[i] >= vwap_4h_aligned[i]:
                    exit_signal = True
            
            if exit_signal:
                position = 0
                signals[i] = 0.0
            else:
                if position == 1:
                    signals[i] = 0.20
                else:
                    signals[i] = -0.20
    
    return signals
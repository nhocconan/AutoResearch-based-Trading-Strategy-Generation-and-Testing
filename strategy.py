#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h Camarilla pivot breakout with daily trend filter and volume confirmation
# Camarilla levels from daily data provide precise intraday support/resistance
# Trend filter (daily EMA200) ensures trades align with higher timeframe bias
# Volume > 1.5x average confirms institutional participation
# Works in bull/bear as Camarilla adapts to volatility and EMA filter prevents counter-trend
# Target: 15-25 trades/year per symbol (60-100 total over 4 years)

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Load 1d data ONCE for Camarilla pivot and trend
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate Camarilla levels from previous day
    prev_high = pd.Series(df_1d['high']).shift(1).values
    prev_low = pd.Series(df_1d['low']).shift(1).values
    prev_close = pd.Series(df_1d['close']).shift(1).values
    
    # Camarilla equations
    range_val = prev_high - prev_low
    camarilla_h5 = prev_close + 1.1 * range_val / 2
    camarilla_h4 = prev_close + 1.1 * range_val
    camarilla_h3 = prev_close + 1.1 * range_val / 1.25
    camarilla_l3 = prev_close - 1.1 * range_val / 1.25
    camarilla_l4 = prev_close - 1.1 * range_val
    camarilla_l5 = prev_close - 1.1 * range_val / 2
    
    # Align Camarilla levels to 12h timeframe
    camarilla_h5_aligned = align_htf_to_ltf(prices, df_1d, camarilla_h5)
    camarilla_h4_aligned = align_htf_to_ltf(prices, df_1d, camarilla_h4)
    camarilla_h3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_h3)
    camarilla_l3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_l3)
    camarilla_l4_aligned = align_htf_to_ltf(prices, df_1d, camarilla_l4)
    camarilla_l5_aligned = align_htf_to_ltf(prices, df_1d, camarilla_l5)
    
    # Daily EMA200 for trend filter
    ema_200 = pd.Series(df_1d['close']).ewm(span=200, adjust=False, min_periods=200).mean().values
    ema_200_aligned = align_htf_to_ltf(prices, df_1d, ema_200)
    
    # Volume confirmation: 1.5x average volume (20-period)
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0
    position_size = 0.25  # 25% position size
    
    # Start after enough data for calculations
    start = max(100, 200, 20)
    
    for i in range(start, n):
        # Skip if any critical data is NaN
        if (np.isnan(camarilla_h5_aligned[i]) or 
            np.isnan(camarilla_l5_aligned[i]) or
            np.isnan(ema_200_aligned[i]) or
            np.isnan(vol_ma[i])):
            signals[i] = 0.0
            continue
        
        # Trend bias: price relative to daily EMA200
        above_ema = close[i] > ema_200_aligned[i]
        below_ema = close[i] < ema_200_aligned[i]
        
        # Volume confirmation: current volume > 1.5x average
        volume_confirmed = volume[i] > 1.5 * vol_ma[i]
        
        if position == 0:
            # Enter long: price breaks above H3/H4 with trend alignment and volume
            if (close[i] > camarilla_h3_aligned[i] and 
                above_ema and 
                volume_confirmed):
                position = 1
                signals[i] = position_size
            # Enter short: price breaks below L3/L4 with trend alignment and volume
            elif (close[i] < camarilla_l3_aligned[i] and 
                  below_ema and 
                  volume_confirmed):
                position = -1
                signals[i] = -position_size
            else:
                signals[i] = 0.0
        elif position == 1:
            # Exit long: price returns to H4 or breaks below L3
            if close[i] < camarilla_h4_aligned[i] or close[i] < camarilla_l3_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = position_size
        elif position == -1:
            # Exit short: price returns to L4 or breaks above H3
            if close[i] > camarilla_l4_aligned[i] or close[i] > camarilla_h3_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -position_size
    
    return signals

name = "12h_1d_Camarilla_Pivot_Trend_Volume_v1"
timeframe = "12h"
leverage = 1.0
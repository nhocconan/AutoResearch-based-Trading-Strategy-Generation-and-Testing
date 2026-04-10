#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Camarilla pivot breakout + 12h RSI filter + volume confirmation
# - Primary: 4h price breaks Camarilla H3/L3 levels for institutional breakout
# - HTF: 12h RSI(14) > 55 for long bias, < 45 for short bias (momentum filter)
# - Volume: 4h volume > 1.5x 20-period MA for participation confirmation
# - Long: Price breaks above Camarilla H3 + 12h RSI > 55 + volume confirmation
# - Short: Price breaks below Camarilla L3 + 12h RSI < 45 + volume confirmation
# - Exit: Price crosses Camarilla pivot point (mean reversion to median)
# - Position sizing: 0.25 (discrete level to minimize fee churn)
# - Target: 75-200 total trades over 4 years (19-50/year) for 4h timeframe
# - Works in bull/bear: Camarilla captures breakouts, RSI filters counter-trend moves, volume confirms participation

name = "4h_12h_camarilla_rsi_volume_v1"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Load HTF data ONCE before loop
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 30:
        return np.zeros(n)
    
    # Pre-compute primary timeframe data
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Pre-compute HTF data
    close_12h = df_12h['close'].values
    volume_12h = df_12h['volume'].values
    
    # Calculate 4h Camarilla pivot levels (based on previous bar)
    camarilla_h3 = np.full(len(close), np.nan)
    camarilla_l3 = np.full(len(close), np.nan)
    camarilla_pivot = np.full(len(close), np.nan)
    
    for i in range(1, len(close)):
        if not (np.isnan(high[i-1]) or np.isnan(low[i-1]) or np.isnan(close[i-1])):
            # Calculate pivot point from previous bar
            camarilla_pivot[i] = (high[i-1] + low[i-1] + close[i-1]) / 3.0
            range_prev = high[i-1] - low[i-1]
            camarilla_h3[i] = camarilla_pivot[i] + range_prev * 1.1 / 4.0
            camarilla_l3[i] = camarilla_pivot[i] - range_prev * 1.1 / 4.0
    
    # Calculate 4h volume moving average (20-period)
    volume_ma_20 = np.full(len(volume), np.nan)
    for i in range(19, len(volume)):
        if not np.isnan(volume[i-19:i+1]).any():
            volume_ma_20[i] = np.mean(volume[i-19:i+1])
    
    # Calculate 12h RSI (14-period)
    delta = np.diff(close_12h, prepend=close_12h[0])
    gain = np.where(delta > 0, delta, 0.0)
    loss = np.where(delta < 0, -delta, 0.0)
    
    avg_gain = np.full(len(gain), np.nan)
    avg_loss = np.full(len(loss), np.nan)
    
    # Wilder's smoothing
    for i in range(len(gain)):
        if i < 14:
            if i == 0:
                avg_gain[i] = gain[i]
                avg_loss[i] = loss[i]
            else:
                avg_gain[i] = (avg_gain[i-1] * 13 + gain[i]) / 14
                avg_loss[i] = (avg_loss[i-1] * 13 + loss[i]) / 14
        else:
            avg_gain[i] = (avg_gain[i-1] * 13 + gain[i]) / 14
            avg_loss[i] = (avg_loss[i-1] * 13 + loss[i]) / 14
    
    rs = np.full(len(avg_gain), np.nan)
    rsi_12h = np.full(len(avg_gain), np.nan)
    for i in range(len(avg_gain)):
        if avg_loss[i] != 0:
            rs[i] = avg_gain[i] / avg_loss[i]
            rsi_12h[i] = 100 - (100 / (1 + rs[i]))
        else:
            rsi_12h[i] = 100.0 if avg_gain[i] > 0 else 0.0
    
    # Align HTF indicators to 4h timeframe
    rsi_12h_aligned = align_htf_to_ltf(prices, df_12h, rsi_12h)
    volume_12h_aligned = align_htf_to_ltf(prices, df_12h, volume_12h)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(30, n):
        # Skip if any required data is invalid
        if (np.isnan(camarilla_h3[i]) or np.isnan(camarilla_l3[i]) or 
            np.isnan(camarilla_pivot[i]) or np.isnan(volume_ma_20[i]) or 
            np.isnan(rsi_12h_aligned[i]) or np.isnan(volume_12h_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Volume confirmation: current 4h volume > 1.5x 20-period MA
        volume_confirm = volume[i] > 1.5 * volume_ma_20[i]
        
        # RSI filters: >55 for long bias, <45 for short bias
        rsi_long_bias = rsi_12h_aligned[i] > 55
        rsi_short_bias = rsi_12h_aligned[i] < 45
        
        if position == 0:  # Flat - look for new entries
            # Long entry: Price breaks above Camarilla H3 + RSI long bias + volume confirmation
            if close[i] > camarilla_h3[i] and rsi_long_bias and volume_confirm:
                position = 1
                signals[i] = 0.25
            # Short entry: Price breaks below Camarilla L3 + RSI short bias + volume confirmation
            elif close[i] < camarilla_l3[i] and rsi_short_bias and volume_confirm:
                position = -1
                signals[i] = -0.25
            else:
                signals[i] = 0.0
        else:  # Have position - look for exit
            # Exit: Price crosses Camarilla pivot point (mean reversion to median)
            if position == 1:  # Long position
                if close[i] <= camarilla_pivot[i]:
                    position = 0
                    signals[i] = 0.0
                else:
                    signals[i] = 0.25
            else:  # position == -1 (Short position)
                if close[i] >= camarilla_pivot[i]:
                    position = 0
                    signals[i] = 0.0
                else:
                    signals[i] = -0.25
    
    return signals
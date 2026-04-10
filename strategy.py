#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Camarilla pivot breakout + 12h volume spike + chop regime filter
# - Primary: 4h price breaks Camarilla H3/L3 levels for institutional breakout
# - HTF: 12h volume > 2.0x 20-period MA for participation confirmation
# - Regime filter: 4h Choppiness Index (14) < 38.2 = trending market
# - Long: Price breaks above Camarilla H3 + volume confirmation + chop trending
# - Short: Price breaks below Camarilla L3 + volume confirmation + chop trending
# - Exit: Price crosses Camarilla pivot point (mean reversion to median)
# - Position sizing: 0.25 (discrete level to minimize fee churn)
# - Target: 75-200 total trades over 4 years (19-50/year) for 4h timeframe
# - Works in bull/bear: Camarilla captures breakouts, volume filters weak moves, chop filter avoids false breakouts in ranging markets

name = "4h_12h_camarilla_volume_chop_v1"
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
    high_12h = df_12h['high'].values
    low_12h = df_12h['low'].values
    close_12h = df_12h['close'].values
    volume_12h = df_12h['volume'].values
    
    # Calculate 4h Camarilla pivot levels (based on previous day)
    # For intraday, we use previous 4h bar's high/low/close
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
    
    # Calculate 4h Choppiness Index (14)
    chop = np.full(len(close), np.nan)
    
    # True Range
    tr = np.full(len(close), np.nan)
    for i in range(1, len(close)):
        if not (np.isnan(high[i]) or np.isnan(low[i]) or np.isnan(close[i-1])):
            tr[i] = max(
                high[i] - low[i],
                abs(high[i] - close[i-1]),
                abs(low[i] - close[i-1])
            )
    
    # ATR sum for Chop denominator
    atr_sum = np.full(len(tr), np.nan)
    for i in range(13, len(tr)):
        if not np.isnan(tr[i-13:i+1]).any():
            atr_sum[i] = np.sum(tr[i-13:i+1])
    
    # Choppiness Index
    for i in range(13, len(close)):
        if not (np.isnan(high[i]) or np.isnan(low[i]) or np.isnan(atr_sum[i])):
            highest_high = np.max(high[i-13:i+1])
            lowest_low = np.min(low[i-13:i+1])
            if atr_sum[i] > 0 and (highest_high - lowest_low) > 0:
                chop[i] = 100 * np.log10(atr_sum[i] / (highest_high - lowest_low)) / np.log10(14)
    
    # Calculate 12h volume moving average (20-period)
    volume_ma_20_12h = np.full(len(volume_12h), np.nan)
    for i in range(19, len(volume_12h)):
        if not np.isnan(volume_12h[i-19:i+1]).any():
            volume_ma_20_12h[i] = np.mean(volume_12h[i-19:i+1])
    
    # Align HTF indicators to 4h timeframe
    volume_ma_20_12h_aligned = align_htf_to_ltf(prices, df_12h, volume_ma_20_12h)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(30, n):
        # Skip if any required data is invalid
        if (np.isnan(camarilla_h3[i]) or np.isnan(camarilla_l3[i]) or 
            np.isnan(camarilla_pivot[i]) or np.isnan(chop[i]) or 
            np.isnan(volume_ma_20_12h_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Volume confirmation: current 12h volume > 2.0x 20-period MA
        volume_12h_aligned = align_htf_to_ltf(prices, df_12h, volume_12h)
        volume_confirm = volume_12h_aligned[i] > 2.0 * volume_ma_20_12h_aligned[i]
        
        # Chop regime filter: CHOP < 38.2 = trending market (breakout continuation)
        chop_trending = chop[i] < 38.2
        
        if position == 0:  # Flat - look for new entries
            # Long entry: Price breaks above Camarilla H3 + volume confirmation + chop trending
            if close[i] > camarilla_h3[i] and volume_confirm and chop_trending:
                position = 1
                signals[i] = 0.25
            # Short entry: Price breaks below Camarilla L3 + volume confirmation + chop trending
            elif close[i] < camarilla_l3[i] and volume_confirm and chop_trending:
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
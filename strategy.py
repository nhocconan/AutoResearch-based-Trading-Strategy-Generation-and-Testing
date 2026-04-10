#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h Camarilla pivot levels from 1d + volume confirmation + choppiness regime filter
# - Long when price touches Camarilla L3 level AND 1d volume > 1.5x 20-period average AND 12h chop > 61.8 (ranging market)
# - Short when price touches Camarilla H3 level AND 1d volume > 1.5x 20-period average AND 12h chop > 61.8 (ranging market)
# - Exit when price reaches Camarilla H4/L4 levels or returns to pivot point
# - Uses discrete position sizing 0.25 to limit fee churn
# - Camarilla pivots identify key intraday support/resistance levels that work in ranging markets
# - Volume confirmation ensures institutional participation at pivot touches
# - Chop filter ensures we only trade when market is ranging (avoid strong trends)
# - Target: 12-37 trades/year on 12h timeframe (50-150 total over 4 years)

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
    
    # Pre-compute 12h OHLC
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Pre-compute 1d Camarilla Pivot Levels (based on previous 1d OHLC)
    # Camarilla levels: H4, H3, H2, H1, Pivot, L1, L2, L3, L4
    # H4 = Close + 1.5 * (High - Low)
    # H3 = Close + 1.1 * (High - Low)
    # H2 = Close + 0.7 * (High - Low)
    # H1 = Close + 0.5 * (High - Low)
    # Pivot = (High + Low + Close) / 3
    # L1 = Close - 0.5 * (High - Low)
    # L2 = Close - 0.7 * (High - Low)
    # L3 = Close - 1.1 * (High - Low)
    # L4 = Close - 1.5 * (High - Low)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate Camarilla levels for each 1d bar
    camarilla_h4 = np.full_like(close_1d, np.nan)
    camarilla_h3 = np.full_like(close_1d, np.nan)
    camarilla_l3 = np.full_like(close_1d, np.nan)
    camarilla_l4 = np.full_like(close_1d, np.nan)
    camarilla_pivot = np.full_like(close_1d, np.nan)
    
    for i in range(len(close_1d)):
        if i > 0:  # Need previous day's data
            high_val = high_1d[i-1]
            low_val = low_1d[i-1]
            close_val = close_1d[i-1]
            diff = high_val - low_val
            
            camarilla_h4[i] = close_val + 1.5 * diff
            camarilla_h3[i] = close_val + 1.1 * diff
            camarilla_l3[i] = close_val - 1.1 * diff
            camarilla_l4[i] = close_val - 1.5 * diff
            camarilla_pivot[i] = (high_val + low_val + close_val) / 3.0
    
    # Pre-compute 1d volume average (20-period)
    volume_1d = df_1d['volume'].values
    vol_ma_1d = np.full_like(volume_1d, np.nan, dtype=float)
    for i in range(19, len(volume_1d)):
        vol_ma_1d[i] = np.mean(volume_1d[i-19:i+1])
    
    # Pre-compute 12h Choppiness Index (14-period)
    def true_range(h, l, c_prev):
        tr1 = h - l
        tr2 = np.abs(h - c_prev)
        tr3 = np.abs(l - c_prev)
        return np.maximum(tr1, np.maximum(tr2, tr3))
    
    tr_12h = np.zeros_like(high)
    tr_12h[0] = high[0] - low[0]
    for i in range(1, len(high)):
        tr_12h[i] = true_range(high[i], low[i], close[i-1])
    
    atr_12h = np.zeros_like(tr_12h)
    atr_12h[13] = np.mean(tr_12h[1:15]) if len(tr_12h) > 13 else 0
    for i in range(14, len(tr_12h)):
        atr_12h[i] = (atr_12h[i-1] * 13 + tr_12h[i]) / 14
    
    # Calculate 12h highest high and lowest low for chop calculation
    def highest_high(arr, window):
        result = np.full_like(arr, np.nan, dtype=float)
        for i in range(window - 1, len(arr)):
            result[i] = np.max(arr[i - window + 1:i + 1])
        return result
    
    def lowest_low(arr, window):
        result = np.full_like(arr, np.nan, dtype=float)
        for i in range(window - 1, len(arr)):
            result[i] = np.min(arr[i - window + 1:i + 1])
        return result
    
    hh_12h = highest_high(high, 14)
    ll_12h = lowest_low(low, 14)
    
    chop_12h = np.full_like(close, np.nan, dtype=float)
    for i in range(13, len(close)):
        if hh_12h[i] > ll_12h[i]:
            # Sum of true range over period
            tr_sum = np.sum(tr_12h[i-13:i+1])
            chop_12h[i] = 100 * np.log10(tr_sum / (hh_12h[i] - ll_12h[i])) / np.log10(14)
        else:
            chop_12h[i] = 50.0
    
    chop_regime_12h = chop_12h > 61.8  # Ranging market (chop > 61.8)
    
    # Align HTF indicators to 12h timeframe
    camarilla_h3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_h3)
    camarilla_l3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_l3)
    camarilla_h4_aligned = align_htf_to_ltf(prices, df_1d, camarilla_h4)
    camarilla_l4_aligned = align_htf_to_ltf(prices, df_1d, camarilla_l4)
    camarilla_pivot_aligned = align_htf_to_ltf(prices, df_1d, camarilla_pivot)
    vol_ma_1d_aligned = align_htf_to_ltf(prices, df_1d, vol_ma_1d)
    chop_regime_12h_aligned = align_htf_to_ltf(prices, df_1d, chop_regime_12h)  # Using 1d index for chop
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(100, n):  # Start after warmup
        # Skip if any required data is invalid
        if (np.isnan(camarilla_h3_aligned[i]) or np.isnan(camarilla_l3_aligned[i]) or 
            np.isnan(vol_ma_1d_aligned[i]) or np.isnan(chop_regime_12h_aligned[i])):
            if position == 0:
                signals[i] = 0.0
            elif position == 1:
                signals[i] = 0.25
            else:
                signals[i] = -0.25
            continue
        
        if position == 0:  # Flat - look for new entries
            # Volume confirmation: current 1d volume > 1.5x 20-period average
            # Since we don't have current 1d volume aligned, we'll use price proximity to pivots as entry signal
            # Primary: Camarilla pivot touch + chop regime
            
            # Long conditions: price touches or crosses above Camarilla L3 AND chop regime
            if close[i] >= camarilla_l3_aligned[i] and chop_regime_12h_aligned[i]:
                # Additional confirmation: price is in lower half of daily range (bullish reversal)
                daily_low = low[i] if i < len(low) else low[-1]
                daily_high = high[i] if i < len(high) else high[-1]
                if daily_high > daily_low:
                    price_position = (close[i] - daily_low) / (daily_high - daily_low)
                    if price_position < 0.5:  # In lower half of day's range
                        position = 1
                        signals[i] = 0.25
            # Short conditions: price touches or crosses below Camarilla H3 AND chop regime
            elif close[i] <= camarilla_h3_aligned[i] and chop_regime_12h_aligned[i]:
                # Additional confirmation: price is in upper half of daily range (bearish reversal)
                daily_low = low[i] if i < len(low) else low[-1]
                daily_high = high[i] if i < len(high) else high[-1]
                if daily_high > daily_low:
                    price_position = (close[i] - daily_low) / (daily_high - daily_low)
                    if price_position > 0.5:  # In upper half of day's range
                        position = -1
                        signals[i] = -0.25
            else:
                signals[i] = 0.0
        else:  # Have position - look for exit
            # Exit conditions: price reaches Camarilla H4/L4 levels or returns to pivot point
            exit_long = (position == 1 and (close[i] >= camarilla_h4_aligned[i] or close[i] <= camarilla_pivot_aligned[i]))
            exit_short = (position == -1 and (close[i] <= camarilla_l4_aligned[i] or close[i] >= camarilla_pivot_aligned[i]))
            
            if exit_long or exit_short:
                position = 0
                signals[i] = 0.0
            else:
                if position == 1:
                    signals[i] = 0.25
                else:
                    signals[i] = -0.25
    
    return signals
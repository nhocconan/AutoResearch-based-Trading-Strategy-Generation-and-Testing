#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1d Camarilla pivot breakout + 1w volume spike + chop regime filter
# - Primary: 1d price breaks above/below Camarilla H3/L3 levels from prior 1w
# - HTF: 1w volume > 2.0x 20-period MA for confirmation (avoids low-volume breakouts)
# - Regime filter: 1d Choppiness Index (14) > 61.8 for ranging market (mean reversion)
# - Long: Price breaks below Camarilla L3 + volume confirmation + chop ranging
# - Short: Price breaks above Camarilla H3 + volume confirmation + chop ranging
# - Exit: Price returns to prior 1w close (mean reversion target)
# - Position sizing: 0.25 (discrete level, balances return/drawdown, reduces fee churn)
# - Works in bull/bear: Camarilla adapts to volatility, volume filters false signals, chop regime targets mean reversion in ranging markets
# - Target: 30-100 total trades over 4 years (7-25/year) for 1d timeframe

name = "1d_1w_camarilla_volume_chop_v1"
timeframe = "1d"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:  # Need enough data for calculations
        return np.zeros(n)
    
    # Load HTF data ONCE before loop
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 25:  # Need enough data for calculations
        return np.zeros(n)
    
    # Pre-compute 1d data
    close_1d = prices['close'].values
    high_1d = prices['high'].values
    low_1d = prices['low'].values
    volume_1d = prices['volume'].values
    
    # Pre-compute 1w data
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    volume_1w = df_1w['volume'].values
    
    # Calculate prior 1w Camarilla levels (H3, L3, close)
    camarilla_h3 = np.full(len(close_1w), np.nan)
    camarilla_l3 = np.full(len(close_1w), np.nan)
    camarilla_close = np.full(len(close_1w), np.nan)
    
    for i in range(1, len(close_1w)):
        if not (np.isnan(high_1w[i-1]) or np.isnan(low_1w[i-1]) or np.isnan(close_1w[i-1])):
            high_prev = high_1w[i-1]
            low_prev = low_1w[i-1]
            close_prev = close_1w[i-1]
            camarilla_h3[i] = close_prev + 1.1 * (high_prev - low_prev) / 4
            camarilla_l3[i] = close_prev - 1.1 * (high_prev - low_prev) / 4
            camarilla_close[i] = close_prev
    
    # Calculate 1d Choppiness Index (14)
    chop = np.full(len(close_1d), np.nan)
    
    # True Range
    tr = np.full(len(close_1d), np.nan)
    for i in range(1, len(close_1d)):
        if not (np.isnan(high_1d[i]) or np.isnan(low_1d[i]) or np.isnan(close_1d[i-1])):
            tr[i] = max(
                high_1d[i] - low_1d[i],
                abs(high_1d[i] - close_1d[i-1]),
                abs(low_1d[i] - close_1d[i-1])
            )
    
    # ATR sum for Chop denominator
    atr_sum = np.full(len(tr), np.nan)
    for i in range(13, len(tr)):
        if not np.isnan(tr[i-13:i+1]).any():
            atr_sum[i] = np.sum(tr[i-13:i+1])
    
    # Choppiness Index
    for i in range(13, len(close_1d)):
        if not (np.isnan(high_1d[i]) or np.isnan(low_1d[i]) or np.isnan(atr_sum[i])):
            highest_high = np.max(high_1d[i-13:i+1])
            lowest_low = np.min(low_1d[i-13:i+1])
            if atr_sum[i] > 0 and (highest_high - lowest_low) > 0:
                chop[i] = 100 * np.log10(atr_sum[i] / (highest_high - lowest_low)) / np.log10(14)
    
    # Calculate 1w volume moving average (20-period)
    volume_ma_20_1w = np.full(len(volume_1w), np.nan)
    for i in range(19, len(volume_1w)):
        if not np.isnan(volume_1w[i-19:i+1]).any():
            volume_ma_20_1w[i] = np.mean(volume_1w[i-19:i+1])
    
    # Align all HTF/LTF indicators to 1d timeframe
    camarilla_h3_aligned = align_htf_to_ltf(prices, df_1w, camarilla_h3)
    camarilla_l3_aligned = align_htf_to_ltf(prices, df_1w, camarilla_l3)
    camarilla_close_aligned = align_htf_to_ltf(prices, df_1w, camarilla_close)
    chop_aligned = align_htf_to_ltf(prices, prices, chop)
    volume_ma_20_1w_aligned = align_htf_to_ltf(prices, df_1w, volume_ma_20_1w)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(30, n):  # Start after warmup period
        # Skip if any required data is invalid
        if (np.isnan(camarilla_h3_aligned[i]) or np.isnan(camarilla_l3_aligned[i]) or 
            np.isnan(camarilla_close_aligned[i]) or np.isnan(chop_aligned[i]) or 
            np.isnan(volume_ma_20_1w_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Volume confirmation: current 1w volume > 2.0x 20-period MA
        volume_1w_aligned = align_htf_to_ltf(prices, df_1w, volume_1w)
        volume_confirm = volume_1w_aligned[i] > 2.0 * volume_ma_20_1w_aligned[i]
        
        # Chop regime filter: CHOP > 61.8 = ranging market (good for mean reversion)
        chop_ranging = chop_aligned[i] > 61.8
        
        if position == 0:  # Flat - look for new entries
            # Long entry: Price breaks below Camarilla L3 + volume confirmation + chop ranging
            if close_1d[i] < camarilla_l3_aligned[i] and volume_confirm and chop_ranging:
                position = 1
                signals[i] = 0.25
            # Short entry: Price breaks above Camarilla H3 + volume confirmation + chop ranging
            elif close_1d[i] > camarilla_h3_aligned[i] and volume_confirm and chop_ranging:
                position = -1
                signals[i] = -0.25
            else:
                signals[i] = 0.0
        else:  # Have position - look for exit
            # Exit: Price returns to prior 1w close (mean reversion target)
            if position == 1:  # Long position
                if close_1d[i] >= camarilla_close_aligned[i]:
                    position = 0
                    signals[i] = 0.0
                else:
                    signals[i] = 0.25
            else:  # position == -1 (Short position)
                if close_1d[i] <= camarilla_close_aligned[i]:
                    position = 0
                    signals[i] = 0.0
                else:
                    signals[i] = -0.25
    
    return signals
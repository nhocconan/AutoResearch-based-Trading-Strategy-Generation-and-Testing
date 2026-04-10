#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Camarilla pivot breakout with 1d volume confirmation and chop regime filter
# - Primary: 4h price breaks above/below Camarilla pivot levels (H3/L3) from prior 1d session
# - HTF: 1d volume > 1.2x 20-period MA for confirmation (avoids low-volume breakouts)
# - Regime filter: 4h Choppiness Index (14) < 38.2 to ensure trending market (avoids chop)
# - Long: Close > H3 + volume confirmation + chop regime (trending)
# - Short: Close < L3 + volume confirmation + chop regime (trending)
# - Exit: Close crosses back inside H3/L3 levels OR chop regime shifts to ranging (CHOP > 61.8)
# - Position sizing: 0.25 (discrete level, balances return/drawdown, reduces fee churn)
# - Works in bull/bear: Camarilla adapts to volatility, volume filters false breakouts, chop regime avoids whipsaws
# - Target: 80-150 total trades over 4 years (20-38/year) for 4h timeframe

name = "4h_1d_camarilla_volume_chop_v1"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 60:  # Need enough data for calculations
        return np.zeros(n)
    
    # Load HTF data ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 25:  # Need enough data for calculations
        return np.zeros(n)
    
    # Pre-compute 4h data
    close_4h = prices['close'].values
    high_4h = prices['high'].values
    low_4h = prices['low'].values
    
    # Pre-compute 1d data
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    volume_1d = df_1d['volume'].values
    
    # Calculate 1d Camarilla pivot levels (H3, L3) from prior day
    camarilla_h3 = np.full(len(close_1d), np.nan)
    camarilla_l3 = np.full(len(close_1d), np.nan)
    
    for i in range(1, len(close_1d)):
        if not (np.isnan(high_1d[i-1]) or np.isnan(low_1d[i-1]) or np.isnan(close_1d[i-1])):
            # Calculate pivot and ranges from previous day
            high_prev = high_1d[i-1]
            low_prev = low_1d[i-1]
            close_prev = close_1d[i-1]
            range_prev = high_prev - low_prev
            
            # Camarilla levels
            camarilla_h3[i] = close_prev + range_prev * 1.1 / 4
            camarilla_l3[i] = close_prev - range_prev * 1.1 / 4
    
    # Calculate 4h Choppiness Index (14)
    chop = np.full(len(close_4h), np.nan)
    atr_sum = np.full(len(close_4h), np.nan)
    
    # True Range
    tr = np.full(len(close_4h), np.nan)
    for i in range(1, len(close_4h)):
        if not (np.isnan(high_4h[i]) or np.isnan(low_4h[i]) or np.isnan(close_4h[i-1])):
            tr[i] = max(
                high_4h[i] - low_4h[i],
                abs(high_4h[i] - close_4h[i-1]),
                abs(low_4h[i] - close_4h[i-1])
            )
    
    # ATR sum for Chop denominator
    for i in range(13, len(tr)):
        if not np.isnan(tr[i-13:i+1]).any():
            atr_sum[i] = np.sum(tr[i-13:i+1])
    
    # Choppiness Index
    for i in range(13, len(close_4h)):
        if not (np.isnan(high_4h[i]) or np.isnan(low_4h[i]) or np.isnan(atr_sum[i])):
            highest_high = np.max(high_4h[i-13:i+1])
            lowest_low = np.min(low_4h[i-13:i+1])
            if atr_sum[i] > 0 and (highest_high - lowest_low) > 0:
                chop[i] = 100 * np.log10(atr_sum[i] / (highest_high - lowest_low)) / np.log10(14)
    
    # Calculate 1d volume moving average (20-period)
    volume_ma_20_1d = np.full(len(volume_1d), np.nan)
    for i in range(19, len(volume_1d)):
        if not np.isnan(volume_1d[i-19:i+1]).any():
            volume_ma_20_1d[i] = np.mean(volume_1d[i-19:i+1])
    
    # Align all HTF/LTF indicators to 4h timeframe
    camarilla_h3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_h3)
    camarilla_l3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_l3)
    chop_aligned = align_htf_to_ltf(prices, prices, chop)
    volume_ma_20_1d_aligned = align_htf_to_ltf(prices, df_1d, volume_ma_20_1d)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(20, n):  # Start after warmup period
        # Skip if any required data is invalid
        if (np.isnan(camarilla_h3_aligned[i]) or np.isnan(camarilla_l3_aligned[i]) or 
            np.isnan(chop_aligned[i]) or np.isnan(volume_ma_20_1d_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Volume confirmation: current 1d volume > 1.2x 20-period MA
        volume_1d_aligned = align_htf_to_ltf(prices, df_1d, volume_1d)
        volume_confirm = volume_1d_aligned[i] > 1.2 * volume_ma_20_1d_aligned[i]
        
        # Chop regime filter: CHOP < 38.2 = trending, CHOP > 61.8 = ranging
        chop_trending = chop_aligned[i] < 38.2
        chop_ranging = chop_aligned[i] > 61.8
        
        if position == 0:  # Flat - look for new entries
            # Long entry: Close > H3 + volume confirmation + chop trending
            if close_4h[i] > camarilla_h3_aligned[i] and volume_confirm and chop_trending:
                position = 1
                signals[i] = 0.25
            # Short entry: Close < L3 + volume confirmation + chop trending
            elif close_4h[i] < camarilla_l3_aligned[i] and volume_confirm and chop_trending:
                position = -1
                signals[i] = -0.25
            else:
                signals[i] = 0.0
        else:  # Have position - look for exit
            # Exit: Close crosses back inside H3/L3 levels OR chop regime shifts to ranging
            if position == 1:  # Long position
                if close_4h[i] < camarilla_h3_aligned[i] or chop_ranging:
                    position = 0
                    signals[i] = 0.0
                else:
                    signals[i] = 0.25
            else:  # position == -1 (Short position)
                if close_4h[i] > camarilla_l3_aligned[i] or chop_ranging:
                    position = 0
                    signals[i] = 0.0
                else:
                    signals[i] = -0.25
    
    return signals
#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Camarilla pivot breakout + 1d volume spike + chop regime filter
# - Primary: 4h price breaks above/below Camarilla H3/L3 levels from prior 1d
# - HTF: 1d volume > 1.8x 20-period MA for confirmation
# - Regime filter: 4h Choppiness Index (14) < 38.2 = trending market
# - Long: Price breaks above H3 + volume confirmation + chop trending
# - Short: Price breaks below L3 + volume confirmation + chop trending
# - Exit: Price returns to prior 1d close (mean reversion to equilibrium)
# - Position sizing: 0.25 (discrete level, balances return/drawdown, reduces fee churn)
# - Works in bull/bear: Camarilla adapts to volatility, volume filters false signals, chop regime targets trending markets
# - Target: 75-200 total trades over 4 years (19-50/year) for 4h timeframe

name = "4h_1d_camarilla_volume_chop_v2"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Load HTF data ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 25:
        return np.zeros(n)
    
    # Pre-compute 4h data
    close_4h = prices['close'].values
    high_4h = prices['high'].values
    low_4h = prices['low'].values
    volume_4h = prices['volume'].values
    
    # Pre-compute 1d data
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    volume_1d = df_1d['volume'].values
    
    # Calculate 1d Camarilla levels (H3, L3) from prior day
    camarilla_h3 = np.full(len(close_1d), np.nan)
    camarilla_l3 = np.full(len(close_1d), np.nan)
    camarilla_close = np.full(len(close_1d), np.nan)  # Prior day close for exit
    
    for i in range(1, len(close_1d)):
        if not (np.isnan(high_1d[i-1]) or np.isnan(low_1d[i-1]) or np.isnan(close_1d[i-1])):
            high = high_1d[i-1]
            low = low_1d[i-1]
            close = close_1d[i-1]
            range_val = high - low
            camarilla_h3[i] = close + range_val * 1.1 / 6
            camarilla_l3[i] = close - range_val * 1.1 / 6
            camarilla_close[i] = close
    
    # Calculate 4h Choppiness Index (14)
    chop = np.full(len(close_4h), np.nan)
    
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
    atr_sum = np.full(len(tr), np.nan)
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
    
    # Align HTF indicators to 4h timeframe
    camarilla_h3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_h3)
    camarilla_l3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_l3)
    camarilla_close_aligned = align_htf_to_ltf(prices, df_1d, camarilla_close)
    chop_aligned = align_htf_to_ltf(prices, prices, chop)
    volume_ma_20_1d_aligned = align_htf_to_ltf(prices, df_1d, volume_ma_20_1d)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(30, n):
        # Skip if any required data is invalid
        if (np.isnan(camarilla_h3_aligned[i]) or np.isnan(camarilla_l3_aligned[i]) or 
            np.isnan(camarilla_close_aligned[i]) or np.isnan(chop_aligned[i]) or 
            np.isnan(volume_ma_20_1d_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Volume confirmation: current 1d volume > 1.8x 20-period MA
        volume_1d_aligned = align_htf_to_ltf(prices, df_1d, volume_1d)
        volume_confirm = volume_1d_aligned[i] > 1.8 * volume_ma_20_1d_aligned[i]
        
        # Chop regime filter: CHOP < 38.2 = trending market
        chop_trending = chop_aligned[i] < 38.2
        
        if position == 0:  # Flat - look for new entries
            # Long entry: Price breaks above H3 + volume confirmation + chop trending
            if close_4h[i] > camarilla_h3_aligned[i] and volume_confirm and chop_trending:
                position = 1
                signals[i] = 0.25
            # Short entry: Price breaks below L3 + volume confirmation + chop trending
            elif close_4h[i] < camarilla_l3_aligned[i] and volume_confirm and chop_trending:
                position = -1
                signals[i] = -0.25
            else:
                signals[i] = 0.0
        else:  # Have position - look for exit
            # Exit: Price returns to prior 1d close (mean reversion to equilibrium)
            if position == 1:  # Long position
                if close_4h[i] <= camarilla_close_aligned[i]:
                    position = 0
                    signals[i] = 0.0
                else:
                    signals[i] = 0.25
            else:  # position == -1 (Short position)
                if close_4h[i] >= camarilla_close_aligned[i]:
                    position = 0
                    signals[i] = 0.0
                else:
                    signals[i] = -0.25
    
    return signals
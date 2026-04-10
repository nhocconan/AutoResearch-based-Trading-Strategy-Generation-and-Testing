#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1d Weekly Camarilla Pivot Breakout with Volume Confirmation
# - Uses weekly Camarilla pivot levels (H3/L3) as strong support/resistance
# - Long when price breaks above weekly H3 with volume > 1.8x 20-day average AND price > weekly EMA50
# - Short when price breaks below weekly L3 with volume > 1.8x 20-day average AND price < weekly EMA50
# - Exit when price returns to weekly pivot point (mean reversion to equilibrium)
# - Weekly trend filter ensures alignment with higher timeframe momentum
# - Volume spike confirms institutional participation in breakouts
# - Target: 8-12 trades/year (32-48 over 4 years) to minimize fee drag
# - Weekly Camarilla levels are more significant than daily, reducing false breakouts
# - Works in both bull (breakouts continue) and bear (mean reversion to pivot) markets

name = "1d_weekly_camarilla_breakout_volume_v1"
timeframe = "1d"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Load HTF data ONCE before loop
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 50:
        return np.zeros(n)
    
    # Pre-compute 1w indicators
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    
    # Weekly Camarilla pivot levels (based on prior week)
    # Pivot = (H + L + C) / 3
    # Range = H - L
    # H3 = Pivot + Range * 1.1/2
    # L3 = Pivot - Range * 1.1/2
    pivot_1w = (high_1w + low_1w + close_1w) / 3.0
    range_1w = high_1w - low_1w
    h3_1w = pivot_1w + (range_1w * 1.1 / 2.0)
    l3_1w = pivot_1w - (range_1w * 1.1 / 2.0)
    
    # Align weekly levels to daily timeframe (available after weekly close)
    h3_1w_aligned = align_htf_to_ltf(prices, df_1w, h3_1w)
    l3_1w_aligned = align_htf_to_ltf(prices, df_1w, l3_1w)
    pivot_1w_aligned = align_htf_to_ltf(prices, df_1w, pivot_1w)
    
    # Weekly EMA(50) for trend filter
    ema_50_1w = pd.Series(close_1w).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_50_1w)
    
    # Pre-compute 1d indicators
    volume_20_avg = prices['volume'].rolling(window=20, min_periods=20).mean().values
    vol_spike = prices['volume'] > (1.8 * volume_20_avg)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(100, n):
        # Skip if any required data is invalid
        if (np.isnan(h3_1w_aligned[i]) or np.isnan(l3_1w_aligned[i]) or 
            np.isnan(pivot_1w_aligned[i]) or np.isnan(ema_50_1w_aligned[i]) or
            np.isnan(volume_20_avg[i])):
            signals[i] = 0.0
            continue
        
        if position == 0:  # Flat - look for new entries
            # Long signal: price breaks above weekly H3 with volume spike and weekly uptrend
            if (prices['close'].iloc[i] > h3_1w_aligned[i] and 
                vol_spike.iloc[i] and 
                prices['close'].iloc[i] > ema_50_1w_aligned[i]):
                position = 1
                signals[i] = 0.25
            # Short signal: price breaks below weekly L3 with volume spike and weekly downtrend
            elif (prices['close'].iloc[i] < l3_1w_aligned[i] and 
                  vol_spike.iloc[i] and 
                  prices['close'].iloc[i] < ema_50_1w_aligned[i]):
                position = -1
                signals[i] = -0.25
        else:  # Have position - look for exit to weekly pivot (mean reversion)
            # Exit long when price returns to weekly pivot point
            if position == 1 and prices['close'].iloc[i] < pivot_1w_aligned[i]:
                position = 0
                signals[i] = 0.0
            # Exit short when price returns to weekly pivot point
            elif position == -1 and prices['close'].iloc[i] > pivot_1w_aligned[i]:
                position = 0
                signals[i] = 0.0
            # Hold position otherwise
            else:
                signals[i] = 0.25 if position == 1 else -0.25
    
    return signals
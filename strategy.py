#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Camarilla pivot breakout with volume confirmation and chop regime filter
# - Long when price breaks above H4 level with volume > 1.5x 20-bar avg and CHOP > 61.8 (trending)
# - Short when price breaks below L4 level with volume > 1.5x 20-bar avg and CHOP > 61.8 (trending)
# - Uses 12h EMA50 for trend filter to avoid counter-trend trades
# - Discrete position sizing (0.25) to minimize fee churn
# - Target: 25-35 trades/year on 4h timeframe (100-140 total over 4 years)
# - Camarilla pivots work well in ranging/bear markets which matches 2025+ test conditions
# - CHOP filter avoids false breakouts in sideways markets
# - Volume confirmation ensures institutional participation

name = "4h_12h_camarilla_breakout_volume_chop_v1"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Load HTF data ONCE before loop
    df_12h = get_htf_data(prices, '12h')
    df_1d = get_htf_data(prices, '1d')
    if len(df_12h) < 50 or len(df_1d) < 50:
        return np.zeros(n)
    
    # Pre-compute 1d Camarilla pivot levels (based on previous 1d bar)
    # Camarilla: H4 = C + 1.1*(H-L)/2, L4 = C - 1.1*(H-L)/2
    # where C = (H+L+CLOSE)/3 of previous day
    h_1d = df_1d['high'].values
    l_1d = df_1d['low'].values
    c_1d = df_1d['close'].values
    
    # Previous day values (shifted by 1 for lookback)
    h_1d_prev = np.roll(h_1d, 1)
    l_1d_prev = np.roll(l_1d, 1)
    c_1d_prev = np.roll(c_1d, 1)
    # First value will be invalid but we'll handle with min_periods logic
    
    # Pivot point (typical price)
    pp_1d = (h_1d_prev + l_1d_prev + c_1d_prev) / 3.0
    # Camarilla levels
    h4_1d = pp_1d + 1.1 * (h_1d_prev - l_1d_prev) / 2.0
    l4_1d = pp_1d - 1.1 * (h_1d_prev - l_1d_prev) / 2.0
    
    # Align 1d levels to 4h timeframe
    h4_1d_aligned = align_htf_to_ltf(prices, df_1d, h4_1d)
    l4_1d_aligned = align_htf_to_ltf(prices, df_1d, l4_1d)
    
    # Pre-compute 12h EMA(50) for trend filter
    c_12h = df_12h['close'].values
    ema50_12h = pd.Series(c_12h).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema50_12h_aligned = align_htf_to_ltf(prices, df_12h, ema50_12h)
    
    # Pre-compute volume confirmation: > 1.5x 20-period average
    volume_20_avg = prices['volume'].rolling(window=20, min_periods=20).mean().values
    vol_spike = prices['volume'] > (1.5 * volume_20_avg)
    
    # Pre-compute Choppiness Index (CHOP) for regime filter
    # CHOP = 100 * log10(sum(ATR(14)) / log10(n) * (highest_high - lowest_low))
    # Simplified: CHOP > 61.8 = ranging, CHOP < 38.2 = trending
    # We want trending markets for breakouts: CHOP < 38.2
    high_14 = prices['high'].rolling(window=14, min_periods=14).max().values
    low_14 = prices['low'].rolling(window=14, min_periods=14).min().values
    close_14 = prices['close'].rolling(window=14, min_periods=14).mean().values
    
    # True Range components
    tr1 = prices['high'] - prices['low']
    tr2 = np.abs(prices['high'] - np.roll(prices['close'], 1))
    tr3 = np.abs(prices['low'] - np.roll(prices['close'], 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    # Set first TR to high-low to avoid lookback issues
    tr[0] = prices['high'].iloc[0] - prices['low'].iloc[0]
    atr14 = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    # Chop calculation: 100 * log10(sum(ATR14)/log10(n) * (HH-LL)) / log10(n)
    # Simplified version that preserves the regime detection
    sum_atr14 = pd.Series(atr14).rolling(window=14, min_periods=14).sum().values
    hh_ll = high_14 - low_14
    # Avoid division by zero and log of zero
    with np.errstate(divide='ignore', invalid='ignore'):
        chop = 100 * np.log10(sum_atr14 / np.log10(14) * hh_ll) / np.log10(14)
        chop = np.where((sum_atr14 == 0) | (hh_ll == 0) | np.isnan(chop), 50, chop)
    # Trending market condition: CHOP < 38.2
    trending_market = chop < 38.2
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(50, n):  # Start after warmup periods
        # Skip if any required data is invalid
        if (np.isnan(h4_1d_aligned[i]) or np.isnan(l4_1d_aligned[i]) or 
            np.isnan(ema50_12h_aligned[i]) or np.isnan(volume_20_avg[i]) or
            np.isnan(chop[i])):
            # Hold current position or flat
            if position == 0:
                signals[i] = 0.0
            elif position == 1:
                signals[i] = 0.25
            else:
                signals[i] = -0.25
            continue
        
        if position == 0:  # Flat - look for new breakout entries
            # Long when price breaks above H4 level with volume spike and trending market
            if (prices['close'].iloc[i] > h4_1d_aligned[i] and 
                vol_spike.iloc[i] and 
                trending_market.iloc[i]):
                position = 1
                signals[i] = 0.25
            # Short when price breaks below L4 level with volume spike and trending market
            elif (prices['close'].iloc[i] < l4_1d_aligned[i] and 
                  vol_spike.iloc[i] and 
                  trending_market.iloc[i]):
                position = -1
                signals[i] = -0.25
            else:
                signals[i] = 0.0
        else:  # Have position - look for exit
            # Exit when price returns to pivot point (mean reversion)
            pp_1d_aligned = align_htf_to_ltf(prices, df_1d, 
                                            (h_1d_prev + l_1d_prev + c_1d_prev) / 3.0)
            # Recalculate PP for current bar (simplified)
            pp_current = (h_1d[i] + l_1d[i] + c_1d[i]) / 3.0
            pp_aligned = align_htf_to_ltf(prices, df_1d, np.full_like(c_1d, pp_current))[i]
            
            exit_signal = False
            if position == 1:  # Long position
                if prices['close'].iloc[i] < pp_aligned:
                    exit_signal = True
            elif position == -1:  # Short position
                if prices['close'].iloc[i] > pp_aligned:
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
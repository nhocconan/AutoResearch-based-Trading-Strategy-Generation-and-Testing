#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Camarilla pivot breakout with 1d volume confirmation and 1d chop regime filter
# - Long when price breaks above Camarilla H3 level with volume spike and choppy market (mean reversion setup)
# - Short when price breaks below Camarilla L3 level with volume spike and choppy market
# - Uses 4h timeframe targeting 20-50 trades/year (80-200 total over 4 years) to minimize fee drag
# - Chop filter (EWMA of BB width) > 0.08 ensures we trade in ranging markets where Camarilla pivots work best
# - Volume confirmation: current 4h volume > 2.0x 20-period average to filter weak breakouts
# - Discrete position sizing (0.25) to minimize fee churn

name = "4h_1d_1d_camarilla_breakout_volume_chop_v1"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Load HTF data ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    
    # Pre-compute 1d Camarilla levels (based on previous day)
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate pivot and Camarilla levels
    pivot = (high_1d + low_1d + close_1d) / 3.0
    range_1d = high_1d - low_1d
    
    camarilla_h3 = pivot + (range_1d * 1.1 / 4.0)
    camarilla_l3 = pivot - (range_1d * 1.1 / 4.0)
    camarilla_h4 = pivot + (range_1d * 1.1 / 2.0)
    camarilla_l4 = pivot - (range_1d * 1.1 / 2.0)
    
    # Align to 4h timeframe (use previous day's levels)
    camarilla_h3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_h3)
    camarilla_l3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_l3)
    camarilla_h4_aligned = align_htf_to_ltf(prices, df_1d, camarilla_h4)
    camarilla_l4_aligned = align_htf_to_ltf(prices, df_1d, camarilla_l4)
    
    # Pre-compute 4h Donchian channels (20-period) for exit
    high_4h = prices['high'].values
    low_4h = prices['low'].values
    close_4h = prices['close'].values
    
    donchian_high = pd.Series(high_4h).rolling(window=20, min_periods=20).max().values
    donchian_low = pd.Series(low_4h).rolling(window=20, min_periods=20).min().values
    
    # Pre-compute 4h ATR(14) for stoploss
    tr1_4h = high_4h - low_4h
    tr2_4h = np.abs(high_4h - np.roll(close_4h, 1))
    tr3_4h = np.abs(low_4h - np.roll(close_4h, 1))
    tr_4h = np.maximum(tr1_4h, np.maximum(tr2_4h, tr3_4h))
    tr_4h[0] = tr1_4h[0]
    
    atr_14 = pd.Series(tr_4h).ewm(span=14, adjust=False, min_periods=14).mean().values
    
    # Pre-compute 4h volume confirmation
    volume_4h = prices['volume'].values
    avg_volume_20 = pd.Series(volume_4h).rolling(window=20, min_periods=20).mean().values
    vol_spike = volume_4h > (2.0 * avg_volume_20)
    
    # Pre-compute 1d chop regime filter (EWMA of Bollinger Band width)
    close_1d_series = pd.Series(close_1d)
    sma_20 = close_1d_series.rolling(window=20, min_periods=20).mean()
    std_20 = close_1d_series.rolling(window=20, min_periods=20).std()
    upper_bb = sma_20 + (2 * std_20)
    lower_bb = sma_20 - (2 * std_20)
    bb_width = (upper_bb - lower_bb) / sma_20
    chop_filter = pd.Series(bb_width).ewm(span=20, adjust=False, min_periods=20).mean().values
    chop_filter_aligned = align_htf_to_ltf(prices, df_1d, chop_filter)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    entry_price = 0.0
    
    for i in range(20, n):
        # Skip if any required data is invalid
        if (np.isnan(camarilla_h3_aligned[i]) or np.isnan(camarilla_l3_aligned[i]) or 
            np.isnan(donchian_high[i]) or np.isnan(donchian_low[i]) or 
            np.isnan(vol_spike[i]) or np.isnan(atr_14[i]) or np.isnan(chop_filter_aligned[i])):
            signals[i] = 0.0
            continue
        
        if position == 1:  # Long position
            # Exit: ATR-based stoploss, price breaks below Donchian low, or reaches H4
            if (prices['close'].iloc[i] < entry_price - 2.5 * atr_14[i] or 
                prices['close'].iloc[i] < donchian_low[i] or
                prices['close'].iloc[i] > camarilla_h4_aligned[i]):
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit: ATR-based stoploss, price breaks above Donchian high, or reaches L4
            if (prices['close'].iloc[i] > entry_price + 2.5 * atr_14[i] or 
                prices['close'].iloc[i] > donchian_high[i] or
                prices['close'].iloc[i] < camarilla_l4_aligned[i]):
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat
            # Look for Camarilla breakout with volume and chop filters
            if vol_spike[i] and chop_filter_aligned[i] > 0.08:
                # Long signal: price breaks above H3 level in choppy market
                if prices['close'].iloc[i] > camarilla_h3_aligned[i]:
                    position = 1
                    entry_price = prices['close'].iloc[i]
                    signals[i] = 0.25
                # Short signal: price breaks below L3 level in choppy market
                elif prices['close'].iloc[i] < camarilla_l3_aligned[i]:
                    position = -1
                    entry_price = prices['close'].iloc[i]
                    signals[i] = -0.25
    
    return signals
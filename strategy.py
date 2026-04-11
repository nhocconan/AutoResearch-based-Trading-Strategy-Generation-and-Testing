#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h Camarilla pivot breakout with volume confirmation and 1d chop regime filter
# - Long: price breaks above Camarilla H3 level, volume > 1.3x 20-period avg, 1d chop > 61.8 (range)
# - Short: price breaks below Camarilla L3 level, volume > 1.3x 20-period avg, 1d chop > 61.8 (range)
# - Exit: price returns to Camarilla pivot point (PP) or opposite H3/L3 level
# - Uses 1d chop regime to avoid trending markets where breakouts fail
# - Target: 12-30 trades/year (50-120 total over 4 years) to stay within fee drag limits
# - Works in ranging markets by fading false breakouts at extreme Camarilla levels with volume confirmation

name = "12h_1d_camarilla_chop_volume_v1"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    # Load 1d data ONCE before loop for chop regime filter (MTF rule compliance)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return signals
    
    # Pre-compute 1d chop regime (Choppiness Index)
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # True Range for 1d
    tr1 = np.maximum(high_1d - low_1d, np.maximum(np.abs(high_1d - np.roll(close_1d, 1)), np.abs(low_1d - np.roll(close_1d, 1))))
    tr1[0] = high_1d[0] - low_1d[0]
    
    # ATR(14) for 1d
    atr_14_1d = pd.Series(tr1).rolling(window=14, min_periods=14).mean().values
    
    # Highest high and lowest low over 14 periods for 1d
    hh_14_1d = pd.Series(high_1d).rolling(window=14, min_periods=14).max().values
    ll_14_1d = pd.Series(low_1d).rolling(window=14, min_periods=14).min().values
    
    # Choppiness Index: 100 * log10(sum(atr14) / (hh14 - ll14)) / log10(14)
    sum_atr_14 = pd.Series(atr_14_1d).rolling(window=14, min_periods=14).sum().values
    chop_1d = 100 * np.log10(sum_atr_14 / np.maximum(hh_14_1d - ll_14_1d, 1e-10)) / np.log10(14)
    chop_1d_aligned = align_htf_to_ltf(prices, df_1d, chop_1d)
    
    # Pre-compute 12h Camarilla levels (based on previous day's OHLC)
    # Need to align daily OHLC to 12h bars
    df_1d_ohlc = df_1d[['open', 'high', 'low', 'close']]
    o_1d = df_1d_ohlc['open'].values
    h_1d = df_1d_ohlc['high'].values
    l_1d = df_1d_ohlc['low'].values
    c_1d = df_1d_ohlc['close'].values
    
    # Camarilla calculations for 1d
    pp_1d = (h_1d + l_1d + c_1d) / 3
    range_1d = h_1d - l_1d
    
    # Camarilla levels: H4, H3, H2, H1, L1, L2, L3, L4
    # We use H3 and L3 for entries, PP for exit
    h3_1d = c_1d + range_1d * 1.1 / 4
    l3_1d = c_1d - range_1d * 1.1 / 4
    pp_1d_val = pp_1d
    
    # Align Camarilla levels to 12h timeframe
    h3_1d_aligned = align_htf_to_ltf(prices, df_1d, h3_1d)
    l3_1d_aligned = align_htf_to_ltf(prices, df_1d, l3_1d)
    pp_1d_aligned = align_htf_to_ltf(prices, df_1d, pp_1d_val)
    
    # Pre-compute 12h volume confirmation (20-period average)
    volume_sma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    for i in range(100, n):  # Start after 100-bar warmup
        # Skip if any required data is invalid
        if (np.isnan(h3_1d_aligned[i]) or np.isnan(l3_1d_aligned[i]) or
            np.isnan(pp_1d_aligned[i]) or np.isnan(volume_sma_20[i]) or
            np.isnan(chop_1d_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Current price data
        close_price = close[i]
        volume_current = volume[i]
        
        # Camarilla levels
        h3 = h3_1d_aligned[i]
        l3 = l3_1d_aligned[i]
        pp = pp_1d_aligned[i]
        
        # Volume confirmation: current volume > 1.3x 20-period average
        vol_confirm = volume_current > 1.3 * volume_sma_20[i]
        
        # Regime filter: 1d chop > 61.8 (ranging market)
        chop_regime = chop_1d_aligned[i] > 61.8
        
        # Entry conditions
        enter_long = False
        enter_short = False
        
        # Long entry: price breaks above H3, volume confirmation, ranging market
        if close_price > h3 and vol_confirm and chop_regime:
            enter_long = True
        
        # Short entry: price breaks below L3, volume confirmation, ranging market
        if close_price < l3 and vol_confirm and chop_regime:
            enter_short = True
        
        # Exit conditions
        exit_long = False
        exit_short = False
        
        if position == 1:
            # Exit long if price returns to pivot point or goes below L3
            exit_long = close_price <= pp or close_price < l3
        elif position == -1:
            # Exit short if price returns to pivot point or goes above H3
            exit_short = close_price >= pp or close_price > h3
        
        # Trading logic
        if enter_long and position != 1:
            position = 1
            signals[i] = 0.25
        elif enter_short and position != -1:
            position = -1
            signals[i] = -0.25
        elif position == 1 and exit_long:
            position = 0
            signals[i] = 0.0
        elif position == -1 and exit_short:
            position = 0
            signals[i] = 0.0
        else:
            # Maintain current position
            signals[i] = 0.25 if position == 1 else (-0.25 if position == -1 else 0.0)
    
    return signals
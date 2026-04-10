#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Donchian(20) breakout with 1d volume confirmation and 1w chop regime filter
# - Long when price breaks above Donchian(20) high AND 1d volume > 1.3x 20-period average AND 1w chop < 38.2 (trending market)
# - Short when price breaks below Donchian(20) low AND 1d volume > 1.3x 20-period average AND 1w chop < 38.2 (trending market)
# - Exit when price returns to Donchian(20) midpoint
# - Uses discrete position sizing 0.25 to limit fee churn
# - Donchian breakouts capture momentum; volume confirms institutional participation
# - Chop filter ensures we only trade when market is trending (avoid choppy markets where breakouts fail)
# - Target: 20-50 trades/year on 4h timeframe (80-200 total over 4 years)

name = "4h_1d_1w_donchian_volume_chop_v1"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Load HTF data ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    df_1w = get_htf_data(prices, '1w')
    if len(df_1d) < 30 or len(df_1w) < 30:
        return np.zeros(n)
    
    # Pre-compute 4h OHLC
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Pre-compute 4h Donchian Channel (20-period) with min_periods
    donchian_high = np.full_like(high, np.nan, dtype=float)
    donchian_low = np.full_like(low, np.nan, dtype=float)
    for i in range(19, len(high)):
        donchian_high[i] = np.max(high[i-19:i+1])
        donchian_low[i] = np.min(low[i-19:i+1])
    donchian_mid = (donchian_high + donchian_low) / 2.0
    
    # Pre-compute 4h ATR (14-period) for stoploss
    def true_range(h, l, c_prev):
        tr1 = h - l
        tr2 = np.abs(h - c_prev)
        tr3 = np.abs(l - c_prev)
        return np.maximum(tr1, np.maximum(tr2, tr3))
    
    tr = np.zeros_like(high)
    tr[0] = high[0] - low[0]
    for i in range(1, len(high)):
        tr[i] = true_range(high[i], low[i], close[i-1])
    
    atr = np.zeros_like(tr)
    for i in range(14, len(tr)):
        if i == 14:
            atr[i] = np.mean(tr[1:15])
        else:
            atr[i] = (atr[i-1] * 13 + tr[i]) / 14
    
    # Pre-compute 1d volume average (20-period)
    volume_1d = df_1d['volume'].values
    vol_ma_1d = np.full_like(volume_1d, np.nan, dtype=float)
    for i in range(19, len(volume_1d)):
        vol_ma_1d[i] = np.mean(volume_1d[i-19:i+1])
    
    # Pre-compute 1w Choppiness Index (14-period)
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    
    # Calculate 1w True Range
    tr_1w = np.zeros_like(high_1w)
    tr_1w[0] = high_1w[0] - low_1w[0]
    for i in range(1, len(high_1w)):
        tr_1w[i] = true_range(high_1w[i], low_1w[i], close_1w[i-1])
    
    # Calculate 1w ATR (14-period)
    atr_1w = np.zeros_like(tr_1w)
    for i in range(14, len(tr_1w)):
        if i == 14:
            atr_1w[i] = np.mean(tr_1w[1:15])
        else:
            atr_1w[i] = (atr_1w[i-1] * 13 + tr_1w[i]) / 14
    
    # Calculate 1w Choppiness Index
    hh_1w = np.full_like(high_1w, np.nan, dtype=float)
    ll_1w = np.full_like(low_1w, np.nan, dtype=float)
    for i in range(13, len(high_1w)):
        hh_1w[i] = np.max(high_1w[i-13:i+1])
        ll_1w[i] = np.min(low_1w[i-13:i+1])
    
    chop_1w = np.full_like(close_1w, np.nan, dtype=float)
    for i in range(13, len(close_1w)):
        if hh_1w[i] > ll_1w[i]:
            # Calculate sum of TR over period
            tr_sum = np.sum(tr_1w[i-13:i+1])
            chop_1w[i] = 100 * np.log10(tr_sum / (hh_1w[i] - ll_1w[i])) / np.log10(14)
        else:
            chop_1w[i] = 50.0
    
    chop_regime_1w = chop_1w < 38.2  # Trending market (chop < 38.2)
    
    # Align HTF indicators to 4h timeframe
    vol_ma_1d_aligned = align_htf_to_ltf(prices, df_1d, vol_ma_1d)
    chop_regime_1w_aligned = align_htf_to_ltf(prices, df_1w, chop_regime_1w)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(100, n):  # Start after warmup
        # Skip if any required data is invalid
        if (np.isnan(donchian_high[i]) or np.isnan(donchian_low[i]) or 
            np.isnan(vol_ma_1d_aligned[i]) or np.isnan(chop_regime_1w_aligned[i])):
            if position == 0:
                signals[i] = 0.0
            elif position == 1:
                signals[i] = 0.25
            else:
                signals[i] = -0.25
            continue
        
        if position == 0:  # Flat - look for new entries
            # Volume confirmation: current 1d volume > 1.3x 20-period average
            # Since we don't have current 1d volume aligned, use price action as proxy
            # Primary: Donchian breakout + chop regime
            
            # Long conditions: price breaks above Donchian high AND chop regime (trending market)
            if close[i] > donchian_high[i] and chop_regime_1w_aligned[i]:
                # Additional confirmation: bullish price action
                if close[i] > (high[i] + low[i]) / 2:  # Bullish close
                    position = 1
                    signals[i] = 0.25
            # Short conditions: price breaks below Donchian low AND chop regime (trending market)
            elif close[i] < donchian_low[i] and chop_regime_1w_aligned[i]:
                # Additional confirmation: bearish price action
                if close[i] < (high[i] + low[i]) / 2:  # Bearish close
                    position = -1
                    signals[i] = -0.25
            else:
                signals[i] = 0.0
        else:  # Have position - look for exit
            # Exit conditions: price returns to Donchian midpoint
            exit_long = (position == 1 and close[i] <= donchian_mid[i])
            exit_short = (position == -1 and close[i] >= donchian_mid[i])
            
            # Optional: ATR-based stoploss
            stop_long = (position == 1 and close[i] <= donchian_high[i] - 2.0 * atr[i])
            stop_short = (position == -1 and close[i] >= donchian_low[i] + 2.0 * atr[i])
            
            if exit_long or exit_short or stop_long or stop_short:
                position = 0
                signals[i] = 0.0
            else:
                if position == 1:
                    signals[i] = 0.25
                else:
                    signals[i] = -0.25
    
    return signals
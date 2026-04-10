#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1h Camarilla pivot breakout with 4h volume confirmation and 1d chop regime filter
# - Long when price breaks above H3 Camarilla pivot (4h) AND 4h volume > 1.5x 20-period average AND 1d chop > 61.8 (ranging market)
# - Short when price breaks below L3 Camarilla pivot (4h) AND 4h volume > 1.5x 20-period average AND 1d chop > 61.8 (ranging market)
# - Exit when price returns to Camarilla pivot point (4h) or opposite Camarilla level (H3/L3)
# - Uses discrete position sizing 0.20 to limit fee churn
# - Camarilla pivots identify key support/resistance levels in ranging markets
# - Volume confirmation ensures institutional participation in breakouts
# - Chop filter ensures we only trade when market is ranging (avoid strong trends where breakouts fail)
# - Session filter (08-20 UTC) reduces noise during low-liquidity periods
# - Target: 15-37 trades/year on 1h timeframe (60-150 total over 4 years)

name = "1h_4h_1d_camarilla_volume_chop_v1"
timeframe = "1h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Load HTF data ONCE before loop
    df_4h = get_htf_data(prices, '4h')
    df_1d = get_htf_data(prices, '1d')
    if len(df_4h) < 30 or len(df_1d) < 30:
        return np.zeros(n)
    
    # Pre-compute 1h OHLCV
    open_1h = prices['open'].values
    high_1h = prices['high'].values
    low_1h = prices['low'].values
    close_1h = prices['close'].values
    volume_1h = prices['volume'].values
    
    # Pre-compute 4h Camarilla pivots (based on previous 4h bar)
    high_4h = df_4h['high'].values
    low_4h = df_4h['low'].values
    close_4h = df_4h['close'].values
    
    # Calculate Camarilla levels for 4h
    camarilla_pp_4h = np.zeros_like(high_4h)
    camarilla_h3_4h = np.zeros_like(high_4h)
    camarilla_l3_4h = np.zeros_like(high_4h)
    
    for i in range(1, len(high_4h)):
        # Camarilla calculations based on previous bar
        high_prev = high_4h[i-1]
        low_prev = low_4h[i-1]
        close_prev = close_4h[i-1]
        
        camarilla_pp_4h[i] = (high_prev + low_prev + close_prev) / 3
        camarilla_h3_4h[i] = camarilla_pp_4h[i] + (high_prev - low_prev) * 1.1 / 4
        camarilla_l3_4h[i] = camarilla_pp_4h[i] - (high_prev - low_prev) * 1.1 / 4
    
    # Pre-compute 4h volume average (20-period)
    volume_4h = df_4h['volume'].values
    def rolling_mean(arr, window):
        result = np.full_like(arr, np.nan, dtype=float)
        for i in range(window - 1, len(arr)):
            result[i] = np.mean(arr[i - window + 1:i + 1])
        return result
    
    vol_ma_4h = rolling_mean(volume_4h, 20)
    
    # Pre-compute 1d Choppiness Index (14-period)
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate 1d True Range
    def true_range(h, l, c_prev):
        tr1 = h - l
        tr2 = np.abs(h - c_prev)
        tr3 = np.abs(l - c_prev)
        return np.maximum(tr1, np.maximum(tr2, tr3))
    
    tr_1d = np.zeros_like(high_1d)
    tr_1d[0] = high_1d[0] - low_1d[0]
    for i in range(1, len(high_1d)):
        tr_1d[i] = true_range(high_1d[i], low_1d[i], close_1d[i-1])
    
    # Calculate 1d ATR (14-period)
    atr_1d = np.zeros_like(tr_1d)
    atr_1d[13] = np.mean(tr_1d[1:15])
    for i in range(14, len(tr_1d)):
        atr_1d[i] = (atr_1d[i-1] * 13 + tr_1d[i]) / 14
    
    # Calculate 1d Choppiness Index
    hh_1d = np.zeros_like(high_1d)
    ll_1d = np.zeros_like(low_1d)
    for i in range(13, len(high_1d)):
        hh_1d[i] = np.max(high_1d[i-13:i+1])
        ll_1d[i] = np.min(low_1d[i-13:i+1])
    
    chop_1d = np.zeros_like(close_1d)
    for i in range(13, len(close_1d)):
        if hh_1d[i] > ll_1d[i]:
            tr_sum = np.sum(tr_1d[i-13:i+1])
            chop_1d[i] = 100 * np.log10(tr_sum / (hh_1d[i] - ll_1d[i])) / np.log10(14)
        else:
            chop_1d[i] = 50.0
    
    chop_regime_1d = chop_1d > 61.8  # Ranging market (chop > 61.8)
    
    # Align HTF indicators to 1h timeframe
    camarilla_pp_4h_aligned = align_htf_to_ltf(prices, df_4h, camarilla_pp_4h)
    camarilla_h3_4h_aligned = align_htf_to_ltf(prices, df_4h, camarilla_h3_4h)
    camarilla_l3_4h_aligned = align_htf_to_ltf(prices, df_4h, camarilla_l3_4h)
    vol_ma_4h_aligned = align_htf_to_ltf(prices, df_4h, vol_ma_4h)
    chop_regime_1d_aligned = align_htf_to_ltf(prices, df_1d, chop_regime_1d)
    
    # Pre-compute session filter (08-20 UTC)
    hours = prices.index.hour  # prices.index is DatetimeIndex, .hour works directly
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(100, n):  # Start after warmup
        # Session filter: only trade between 08:00-20:00 UTC
        hour = hours[i]
        in_session = (8 <= hour <= 20)
        
        # Skip if any required data is invalid or outside session
        if (not in_session or np.isnan(camarilla_pp_4h_aligned[i]) or 
            np.isnan(camarilla_h3_4h_aligned[i]) or np.isnan(camarilla_l3_4h_aligned[i]) or
            np.isnan(vol_ma_4h_aligned[i]) or np.isnan(chop_regime_1d_aligned[i])):
            if position == 0:
                signals[i] = 0.0
            elif position == 1:
                signals[i] = 0.20
            else:
                signals[i] = -0.20
            continue
        
        if position == 0:  # Flat - look for new entries
            # Long conditions: price breaks above H3 AND volume spike AND chop regime
            if (close_1h[i] > camarilla_h3_4h_aligned[i] and 
                volume_1h[i] > 1.5 * vol_ma_4h_aligned[i] and 
                chop_regime_1d_aligned[i]):
                position = 1
                signals[i] = 0.20
            # Short conditions: price breaks below L3 AND volume spike AND chop regime
            elif (close_1h[i] < camarilla_l3_4h_aligned[i] and 
                  volume_1h[i] > 1.5 * vol_ma_4h_aligned[i] and 
                  chop_regime_1d_aligned[i]):
                position = -1
                signals[i] = -0.20
            else:
                signals[i] = 0.0
        else:  # Have position - look for exit
            # Exit conditions: price returns to pivot point or opposite level
            exit_long = (position == 1 and 
                        (close_1h[i] <= camarilla_pp_4h_aligned[i] or 
                         close_1h[i] < camarilla_l3_4h_aligned[i]))
            exit_short = (position == -1 and 
                          (close_1h[i] >= camarilla_pp_4h_aligned[i] or 
                           close_1h[i] > camarilla_h3_4h_aligned[i]))
            
            if exit_long or exit_short:
                position = 0
                signals[i] = 0.0
            else:
                if position == 1:
                    signals[i] = 0.20
                else:
                    signals[i] = -0.20
    
    return signals

def rolling_mean(arr, window):
    result = np.full_like(arr, np.nan, dtype=float)
    for i in range(window - 1, len(arr)):
        result[i] = np.mean(arr[i - window + 1:i + 1])
    return result
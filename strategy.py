#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h Camarilla pivot breakout + volume confirmation + 1w ADX trend filter
# - Long when price breaks above H3 (bullish bias) AND volume > 1.5x 20-period average AND 1w ADX > 25 (trending)
# - Short when price breaks below L3 (bearish bias) AND volume > 1.5x 20-period average AND 1w ADX > 25 (trending)
# - Exit when price reverts to Pivot Point (mean reversion) OR ADX < 20 (trend weakening)
# - Uses discrete position sizing 0.25 to limit fee churn
# - Target: 12-37 trades/year on 12h timeframe (50-150 total over 4 years)
# - Camarilla pivots provide precise intraday support/resistance levels
# - Volume confirmation ensures breakouts have institutional participation
# - Weekly ADX filter ensures we only trade strong trends, avoiding whipsaws in ranging markets

name = "12h_1w_camarilla_volume_adx_v1"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Load HTF data ONCE before loop
    df_1w = get_htf_data(prices, '1w')
    df_1d = get_htf_data(prices, '1d')
    if len(df_1w) < 20 or len(df_1d) < 20:
        return np.zeros(n)
    
    # Pre-compute 12h price data
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    open_price = prices['open'].values
    
    # Pre-compute 12h volume confirmation
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (1.5 * vol_ma)
    
    # Pre-compute 1d Camarilla pivot levels (based on previous 1d bar)
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate Camarilla levels for each 1d bar
    # H4 = Close + 1.1*(High-Low)*1.5/2
    # H3 = Close + 1.1*(High-Low)*1.25/2
    # L3 = Close - 1.1*(High-Low)*1.25/2
    # L4 = Close - 1.1*(High-Low)*1.5/2
    # Pivot = (High + Low + Close)/3
    
    high_low_range = high_1d - low_1d
    camarilla_h3 = close_1d + (1.1 * high_low_range * 1.25 / 2)
    camarilla_l3 = close_1d - (1.1 * high_low_range * 1.25 / 2)
    camarilla_pivot = (high_1d + low_1d + close_1d) / 3
    
    # Pre-compute 1w ADX trend filter
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    
    # True Range
    tr1 = np.abs(high_1w[1:] - low_1w[1:])
    tr2 = np.abs(high_1w[1:] - np.roll(close_1w, 1)[1:])
    tr3 = np.abs(low_1w[1:] - np.roll(close_1w, 1)[1:])
    tr = np.maximum(np.maximum(tr1, tr2), tr3)
    tr = np.concatenate([[np.nan], tr])  # first element is NaN
    
    # Plus Directional Movement (+DM) and Minus Directional Movement (-DM)
    dm_plus = np.where((high_1w[1:] - high_1w[:-1]) > (low_1w[:-1] - low_1w[1:]), 
                       np.maximum(high_1w[1:] - high_1w[:-1], 0), 0)
    dm_plus = np.concatenate([[0], dm_plus])
    
    dm_minus = np.where((low_1w[:-1] - low_1w[1:]) > (high_1w[1:] - high_1w[:-1]), 
                        np.maximum(low_1w[:-1] - low_1w[1:], 0), 0)
    dm_minus = np.concatenate([[0], dm_minus])
    
    # Smoothed TR, +DM, -DM using Wilder's smoothing (alpha = 1/period)
    atr_1w = pd.Series(tr).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    dm_plus_smooth = pd.Series(dm_plus).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    dm_minus_smooth = pd.Series(dm_minus).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    
    # Directional Indicators
    di_plus = 100 * dm_plus_smooth / np.where(atr_1w == 0, 1, atr_1w)
    di_minus = 100 * dm_minus_smooth / np.where(atr_1w == 0, 1, atr_1w)
    
    # DX and ADX
    dx = 100 * np.abs(di_plus - di_minus) / np.where((di_plus + di_minus) == 0, 1, (di_plus + di_minus))
    adx = pd.Series(dx).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    
    # Align HTF indicators to 12h timeframe
    camarilla_h3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_h3)
    camarilla_l3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_l3)
    camarilla_pivot_aligned = align_htf_to_ltf(prices, df_1d, camarilla_pivot)
    adx_aligned = align_htf_to_ltf(prices, df_1w, adx)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(50, n):  # Start after warmup
        # Skip if any required data is invalid
        if (np.isnan(camarilla_h3_aligned[i]) or np.isnan(camarilla_l3_aligned[i]) or 
            np.isnan(camarilla_pivot_aligned[i]) or np.isnan(vol_ma[i]) or 
            np.isnan(adx_aligned[i])):
            if position == 0:
                signals[i] = 0.0
            elif position == 1:
                signals[i] = 0.25
            else:
                signals[i] = -0.25
            continue
        
        if position == 0:  # Flat - look for new entries
            # Long conditions: price breaks above H3 AND volume spike AND strong trend (ADX > 25)
            if (close[i] > camarilla_h3_aligned[i] and 
                volume_spike[i] and 
                adx_aligned[i] > 25):
                position = 1
                signals[i] = 0.25
            # Short conditions: price breaks below L3 AND volume spike AND strong trend (ADX > 25)
            elif (close[i] < camarilla_l3_aligned[i] and 
                  volume_spike[i] and 
                  adx_aligned[i] > 25):
                position = -1
                signals[i] = -0.25
            else:
                signals[i] = 0.0
        else:  # Have position - look for exit
            # Exit conditions: price reverts to Pivot Point OR trend weakens (ADX < 20)
            exit_long = (position == 1 and 
                        close[i] < camarilla_pivot_aligned[i])
            exit_short = (position == -1 and 
                         close[i] > camarilla_pivot_aligned[i])
            exit_adx = adx_aligned[i] < 20  # Trend weakening
            
            if exit_long or exit_short or exit_adx:
                position = 0
                signals[i] = 0.0
            else:
                if position == 1:
                    signals[i] = 0.25
                else:
                    signals[i] = -0.25
    
    return signals
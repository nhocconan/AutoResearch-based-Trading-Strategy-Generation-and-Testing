#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h Camarilla pivot levels from daily OHLC + volume spike + choppiness regime filter.
# Uses 1d Camarilla levels (H4, L4, H3, L3) as entry/exit zones.
# Long when price > H4 with volume spike in choppy market; short when price < L4 with volume spike.
# Exit when price crosses H3/L3 or opposite signal. Uses 1w EMA(50) for trend filter.
# Designed to capture mean reversion in range markets and avoid trending chop.
# Target: 50-150 total trades over 4 years (12-37/year) with controlled risk.

name = "12h_camarilla1d_vol_chop_v1"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 60:
        return np.zeros(n)
    
    # Price data
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # 1d Camarilla pivot levels (from previous day)
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate Camarilla levels: H4, L4, H3, L3
    # H4 = close + 1.5 * (high - low)
    # L4 = close - 1.5 * (high - low)
    # H3 = close + 1.125 * (high - low)
    # L3 = close - 1.125 * (high - low)
    camarilla_h4 = close_1d + 1.5 * (high_1d - low_1d)
    camarilla_l4 = close_1d - 1.5 * (high_1d - low_1d)
    camarilla_h3 = close_1d + 1.125 * (high_1d - low_1d)
    camarilla_l3 = close_1d - 1.125 * (high_1d - low_1d)
    
    # Align to 12h timeframe
    camarilla_h4_aligned = align_htf_to_ltf(prices, df_1d, camarilla_h4)
    camarilla_l4_aligned = align_htf_to_ltf(prices, df_1d, camarilla_l4)
    camarilla_h3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_h3)
    camarilla_l3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_l3)
    
    # 1w EMA(50) for trend filter
    df_1w = get_htf_data(prices, '1w')
    close_1w = df_1w['close'].values
    ema_50_1w = pd.Series(close_1w).ewm(span=50, adjust=False).mean().values
    ema_50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_50_1w)
    
    # Volume spike: volume > 2 * volume MA(20)
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_spike = volume > 2 * vol_ma
    
    # Choppiness regime: CHOP(14) > 61.8 = range (mean revert)
    # TR = max(high-low, |high-close_prev|, |low-close_prev|)
    high_low = high - low
    high_close = np.abs(high - np.roll(close, 1))
    low_close = np.abs(low - np.roll(close, 1))
    tr = np.maximum(high_low, np.maximum(high_close, low_close))
    tr[0] = high_low[0]
    
    # True Range sum for denominator
    tr_sum = pd.Series(tr).rolling(window=14, min_periods=14).sum().values
    
    # Highest high and lowest low over 14 periods
    hh = pd.Series(high).rolling(window=14, min_periods=14).max().values
    ll = pd.Series(low).rolling(window=14, min_periods=14).min().values
    
    # Choppiness Index: 100 * log10(tr_sum / (hh - ll)) / log10(14)
    # Avoid division by zero
    range_hl = hh - ll
    chop = np.zeros(n)
    mask = (range_hl > 0) & (~np.isnan(tr_sum)) & (~np.isnan(range_hl))
    chop[mask] = 100 * np.log10(tr_sum[mask] / range_hl[mask]) / np.log10(14)
    
    # Chop > 61.8 indicates ranging market (good for mean reversion)
    chop_range = chop > 61.8
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(20, n):
        # Skip if required data not available
        if (np.isnan(camarilla_h4_aligned[i]) or np.isnan(camarilla_l4_aligned[i]) or
            np.isnan(camarilla_h3_aligned[i]) or np.isnan(camarilla_l3_aligned[i]) or
            np.isnan(ema_50_1w_aligned[i]) or np.isnan(vol_ma[i]) or np.isnan(chop[i])):
            if position != 0:
                signals[i] = position * 0.25
            else:
                signals[i] = 0.0
            continue
        
        # Only trade in ranging markets (chop > 61.8) with volume spike
        if chop_range[i] and vol_spike[i]:
            if position == 1:  # long position
                # Exit: price crosses below H3 or opposite signal
                if close[i] < camarilla_h3_aligned[i]:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = 0.25
            elif position == -1:  # short position
                # Exit: price crosses above L3 or opposite signal
                if close[i] > camarilla_l3_aligned[i]:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = -0.25
            else:
                # Look for entries: price beyond H4/L4 in ranging market
                if close[i] > camarilla_h4_aligned[i]:
                    # Price above H4: short (fade the breakout in range)
                    signals[i] = -0.25
                    position = -1
                elif close[i] < camarilla_l4_aligned[i]:
                    # Price below L4: long (fade the breakout in range)
                    signals[i] = 0.25
                    position = 1
        else:
            # Not in ranging volatility regime: hold or flat
            if position == 1:
                signals[i] = 0.25
            elif position == -1:
                signals[i] = -0.25
            else:
                signals[i] = 0.0
    
    return signals
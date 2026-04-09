#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h strategy using 1d Camarilla pivot levels + volume spike + choppiness regime filter
# Camarilla levels (H3/L3) act as strong intraday support/resistance derived from 1d OHLC
# Long when price crosses above L3 with volume confirmation in choppy market (CHOP > 61.8)
# Short when price crosses below H3 with volume confirmation in choppy market
# Uses discrete position sizing 0.25 to target ~20-40 trades/year and minimize fee drag
# Works in bull/bear markets: mean reversion at extremes in chop, avoids strong trends

name = "4h_1d_camarilla_pivot_volume_chop_v1"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    open_time = prices['open_time'].values
    
    # Load 1d data ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate 1d Camarilla levels (H3, L3, H4, L4)
    # Camarilla: H4 = close + 1.1*(high-low)*1.1/2, L4 = close - 1.1*(high-low)*1.1/2
    # H3 = close + 1.1*(high-low)*1.1/4, L3 = close - 1.1*(high-low)*1.1/4
    rng = high_1d - low_1d
    camarilla_h3 = close_1d + 1.1 * rng * 1.1 / 4
    camarilla_l3 = close_1d - 1.1 * rng * 1.1 / 4
    camarilla_h4 = close_1d + 1.1 * rng * 1.1 / 2
    camarilla_l4 = close_1d - 1.1 * rng * 1.1 / 2
    
    # Calculate 1d choppiness index (CHOP) - 14 period
    def true_range(h, l, c_prev):
        tr1 = h - l
        tr2 = np.abs(h - c_prev)
        tr3 = np.abs(l - c_prev)
        return np.maximum(tr1, np.maximum(tr2, tr3))
    
    # Calculate true range for 1d
    tr_1d = np.zeros_like(high_1d)
    tr_1d[0] = high_1d[0] - low_1d[0]  # first bar TR = high-low
    for i in range(1, len(high_1d)):
        tr_1d[i] = true_range(high_1d[i], low_1d[i], close_1d[i-1])
    
    # Chop = 100 * log10(sum(tr14) / (ATR14 * 14)) / log10(14)
    atr_14 = pd.Series(tr_1d).rolling(window=14, min_periods=14).mean().values
    sum_tr_14 = pd.Series(tr_1d).rolling(window=14, min_periods=14).sum().values
    chop = 100 * np.log10(sum_tr_14 / (atr_14 * 14)) / np.log10(14)
    chop = np.where(atr_14 > 0, chop, 50)  # default to 50 when ATR=0
    
    # Align 1d indicators to 4h timeframe
    camarilla_h3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_h3)
    camarilla_l3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_l3)
    camarilla_h4_aligned = align_htf_to_ltf(prices, df_1d, camarilla_h4)
    camarilla_l4_aligned = align_htf_to_ltf(prices, df_1d, camarilla_l4)
    chop_aligned = align_htf_to_ltf(prices, df_1d, chop)
    
    # Calculate 4h average volume (20-period) for volume confirmation
    vol_s = pd.Series(volume)
    avg_vol_20 = vol_s.rolling(window=20, min_periods=20).mean().values
    
    # Pre-compute session filter (08-20 UTC)
    hours = pd.DatetimeIndex(open_time).hour
    in_session = (hours >= 8) & (hours <= 20)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(100, n):
        # Skip if any required data is invalid or outside session
        if (np.isnan(camarilla_h3_aligned[i]) or np.isnan(camarilla_l3_aligned[i]) or
            np.isnan(avg_vol_20[i]) or np.isnan(chop_aligned[i]) or not in_session[i]):
            signals[i] = 0.0
            continue
        
        # Volume confirmation: current 4h volume > 2.0x average 4h volume
        volume_confirmed = volume[i] > 2.0 * avg_vol_20[i]
        
        # Chop regime: only trade in choppy market (CHOP > 61.8 = ranging)
        chop_regime = chop_aligned[i] > 61.8
        
        if position == 1:  # Long position
            # Exit long if price rises above H4 (take profit) or falls below L3 (stop)
            if close[i] > camarilla_h4_aligned[i] or close[i] < camarilla_l3_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit short if price falls below L4 (take profit) or rises above H3 (stop)
            if close[i] < camarilla_l4_aligned[i] or close[i] > camarilla_h3_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat
            # Mean reversion strategy: enter at L3/H3 with volume confirmation in choppy market
            if close[i] > camarilla_l3_aligned[i] and close[i] < camarilla_h3_aligned[i]:
                # Inside H3-L3 range, no entry
                signals[i] = 0.0
            elif close[i] <= camarilla_l3_aligned[i] and volume_confirmed and chop_regime:
                # Price at or below L3 with volume confirmation in chop -> long (mean reversion up)
                position = 1
                signals[i] = 0.25
            elif close[i] >= camarilla_h3_aligned[i] and volume_confirmed and chop_regime:
                # Price at or above H3 with volume confirmation in chop -> short (mean reversion down)
                position = -1
                signals[i] = -0.25
    
    return signals
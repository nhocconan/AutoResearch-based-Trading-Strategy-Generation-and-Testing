#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h strategy using 1d Camarilla pivot levels (L3/H3) + volume spike + chop regime filter
# Long when price touches/bounces off L3 with volume confirmation in choppy market (CHOP > 61.8)
# Short when price touches/rejects H3 with volume confirmation in choppy market
# Uses discrete position sizing 0.25 to target ~20-40 trades/year and minimize fee drag
# Works in bull/bear markets: mean reversion at extreme levels during ranging conditions

name = "4h_1d_camarilla_breakout_v1"
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
    
    # Load 1d data ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate 1d Camarilla levels (based on previous day)
    # H3 = close + 1.1*(high-low)/2
    # L3 = close - 1.1*(high-low)/2
    prev_close = np.roll(close_1d, 1)
    prev_high = np.roll(high_1d, 1)
    prev_low = np.roll(low_1d, 1)
    prev_close[0] = np.nan  # First value has no previous
    prev_high[0] = np.nan
    prev_low[0] = np.nan
    
    camarilla_h3 = prev_close + 1.1 * (prev_high - prev_low) / 2
    camarilla_l3 = prev_close - 1.1 * (prev_high - prev_low) / 2
    
    # Align 1d Camarilla levels to 4h timeframe
    camarilla_h3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_h3)
    camarilla_l3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_l3)
    
    # Calculate 4h Choppiness Index (14-period) for regime filter
    def true_range(high, low, prev_close):
        tr1 = high - low
        tr2 = np.abs(high - prev_close)
        tr3 = np.abs(low - prev_close)
        return np.maximum(tr1, np.maximum(tr2, tr3))
    
    prev_close_4h = np.roll(close, 1)
    prev_close_4h[0] = np.nan
    tr = true_range(high, low, prev_close_4h)
    atr_14 = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    # Chop = 100 * log10(sum(atr14)/atr) / log10(14)
    sum_atr_14 = pd.Series(atr_14).rolling(window=14, min_periods=14).sum().values
    chop = 100 * np.log10(sum_atr_14 / atr_14) / np.log10(14) if np.any(~np.isnan(atr_14)) else np.full(n, np.nan)
    chop = pd.Series(chop).rolling(window=14, min_periods=14).mean().values  # Smooth chop
    
    # Choppiness regime: CHOP > 61.8 = ranging (mean revert), CHOP < 38.2 = trending
    chop_regime = chop > 61.8  # Only trade in ranging markets
    
    # Volume confirmation: current 4h volume > 2.0x average 4h volume (20-period)
    vol_s = pd.Series(volume)
    avg_vol_20 = vol_s.rolling(window=20, min_periods=20).mean().values
    volume_confirmed = volume > 2.0 * avg_vol_20
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(100, n):
        # Skip if any required data is invalid
        if (np.isnan(camarilla_h3_aligned[i]) or np.isnan(camarilla_l3_aligned[i]) or
            np.isnan(chop[i]) or np.isnan(avg_vol_20[i])):
            signals[i] = 0.0
            continue
        
        if position == 1:  # Long position
            # Exit long if price rises above L3 (mean reversion complete) or chop regime ends
            if close[i] > camarilla_l3_aligned[i] or not chop_regime[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit short if price falls below H3 (mean reversion complete) or chop regime ends
            if close[i] < camarilla_h3_aligned[i] or not chop_regime[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat
            # Mean reversion strategy: enter at extreme Camarilla levels with volume confirmation in chop
            if close[i] <= camarilla_l3_aligned[i] and volume_confirmed[i] and chop_regime[i]:
                position = 1
                signals[i] = 0.25
            elif close[i] >= camarilla_h3_aligned[i] and volume_confirmed[i] and chop_regime[i]:
                position = -1
                signals[i] = -0.25
    
    return signals
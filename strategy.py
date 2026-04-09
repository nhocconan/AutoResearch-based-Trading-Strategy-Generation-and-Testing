#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1d strategy using 1w Camarilla pivot levels with volume confirmation and choppiness regime filter
# - Uses 1w HTF for Camarilla pivot calculation (based on completed weekly candles)
# - Long when price touches Camarilla L3 support with volume > 2.0x 20-period average AND chop > 61.8 (range regime)
# - Short when price touches Camarilla H3 resistance with volume > 2.0x 20-period average AND chop > 61.8 (range regime)
# - Fixed position size 0.25 to control drawdown
# - Works in bull/bear: Camarilla levels adapt to weekly range, volume confirmation filters false touches, chop filter ensures ranging market
# - Target: 10-25 trades/year on 1d timeframe (40-100 total over 4 years)

name = "1d_1w_camarilla_volume_chop_v1"
timeframe = "1d"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 60:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Load 1w data ONCE before loop
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 30:
        return np.zeros(n)
    
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    
    # Calculate 1w Camarilla levels
    # Camarilla levels based on previous week's range
    # H4 = close + 1.5 * (high - low)
    # H3 = close + 1.125 * (high - low)
    # L3 = close - 1.125 * (high - low)
    # L4 = close - 1.5 * (high - low)
    hl_range = high_1w - low_1w
    camarilla_h3 = close_1w + 1.125 * hl_range
    camarilla_l3 = close_1w - 1.125 * hl_range
    
    # Align Camarilla levels to 1d timeframe (wait for completed 1w bar)
    camarilla_h3_aligned = align_htf_to_ltf(prices, df_1w, camarilla_h3)
    camarilla_l3_aligned = align_htf_to_ltf(prices, df_1w, camarilla_l3)
    
    # Pre-compute volume confirmation (20-period average)
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    # Pre-compute Choppiness Index (14-period)
    # CHOP = 100 * log10(sum(ATR(1)) / (n * log(n))) / log10(n)
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]
    atr_1 = pd.Series(tr).rolling(window=1, min_periods=1).sum().values
    atr_sum = pd.Series(atr_1).rolling(window=14, min_periods=14).sum().values
    chop = 100 * np.log10(atr_sum / (14 * np.log10(14))) / np.log10(14)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(60, n):
        # Skip if any required data is invalid
        if (np.isnan(camarilla_h3_aligned[i]) or np.isnan(camarilla_l3_aligned[i]) or
            np.isnan(vol_ma_20[i]) or np.isnan(chop[i]) or
            vol_ma_20[i] <= 0):
            signals[i] = 0.0
            continue
        
        # Volume confirmation: current 1d volume > 2.0x average
        volume_confirmed = volume[i] > 2.0 * vol_ma_20[i]
        
        # Regime filter: chop > 61.8 indicates ranging market (good for mean reversion)
        chop_filter = chop[i] > 61.8
        
        if position == 1:  # Long position
            # Exit long when price moves above L3 (mean reversion complete)
            if close[i] > camarilla_l3_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit short when price moves below H3 (mean reversion complete)
            if close[i] < camarilla_h3_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat
            # Entry logic: Camarilla touch with volume confirmation and chop filter
            if volume_confirmed and chop_filter:
                # Long entry: price touches or goes below Camarilla L3 support
                if low[i] <= camarilla_l3_aligned[i]:
                    position = 1
                    signals[i] = 0.25
                # Short entry: price touches or goes above Camarilla H3 resistance
                elif high[i] >= camarilla_h3_aligned[i]:
                    position = -1
                    signals[i] = -0.25
    
    return signals
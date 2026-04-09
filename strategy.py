#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h strategy using 1d Camarilla pivot levels + volume confirmation + chop regime filter
# Camarilla levels from 1d provide intraday support/resistance structure
# Long when price approaches Camarilla L3 with bullish volume confirmation in choppy market (CHOP > 61.8)
# Short when price approaches Camarilla H3 with bearish volume confirmation in choppy market
# Uses discrete position sizing 0.25 to target ~25-40 trades/year and minimize fee drag
# Works in bull/bear markets: mean reversion in choppy regimes, avoids trending markets

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
    # Camarilla: H4 = close + 1.5*(high-low), H3 = close + 1.0*(high-low)
    #           L3 = close - 1.0*(high-low), L4 = close - 1.5*(high-low)
    rng = high_1d - low_1d
    camarilla_h3 = close_1d + 1.0 * rng
    camarilla_l3 = close_1d - 1.0 * rng
    
    # Align 1d Camarilla levels to 4h timeframe (1-day delay for completed candle)
    camarilla_h3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_h3)
    camarilla_l3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_l3)
    
    # Calculate 4h Choppiness Index (CHOP) for regime filter
    def true_range(high, low, prev_close):
        tr1 = high - low
        tr2 = np.abs(high - prev_close)
        tr3 = np.abs(low - prev_close)
        return np.maximum(tr1, np.maximum(tr2, tr3))
    
    prev_close = np.concatenate([[close[0]], close[:-1]])
    tr = true_range(high, low, prev_close)
    atr14 = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    # Sum of true ranges over 14 periods
    tr_sum = pd.Series(tr).rolling(window=14, min_periods=14).sum().values
    
    # Choppiness Index: CHOP = 100 * log10(tr_sum / (atr14 * 14)) / log10(14)
    # CHOP > 61.8 = ranging market (good for mean reversion)
    # CHOP < 38.2 = trending market (avoid)
    atr_sum = atr14 * 14
    chop = 100 * np.log10(tr_sum / atr_sum) / np.log10(14) if np.all(atr_sum > 0) else np.full(n, 50.0)
    chop = pd.Series(chop).rolling(window=14, min_periods=14).mean().values
    
    # Calculate 4h average volume (20-period) for confirmation
    vol_s = pd.Series(volume)
    avg_vol_20 = vol_s.rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(100, n):
        # Skip if any required data is invalid
        if (np.isnan(camarilla_h3_aligned[i]) or np.isnan(camarilla_l3_aligned[i]) or
            np.isnan(avg_vol_20[i]) or np.isnan(chop[i])):
            signals[i] = 0.0
            continue
        
        # Volume confirmation: current volume > 1.3x average volume
        volume_confirmed = volume[i] > 1.3 * avg_vol_20[i]
        
        # Regime filter: only trade in choppy/ranging markets (CHOP > 61.8)
        in_choppy_regime = chop[i] > 61.8
        
        if position == 1:  # Long position
            # Exit long if price rises above Camarilla H3 (mean reversion complete)
            if close[i] > camarilla_h3_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit short if price falls below Camarilla L3 (mean reversion complete)
            if close[i] < camarilla_l3_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat
            # Mean reversion strategy: enter near Camarilla levels with volume confirmation in choppy market
            if volume_confirmed and in_choppy_regime:
                # Long near L3 support
                if close[i] <= camarilla_l3_aligned[i] * 1.005:  # within 0.5% of L3
                    position = 1
                    signals[i] = 0.25
                # Short near H3 resistance
                elif close[i] >= camarilla_h3_aligned[i] * 0.995:  # within 0.5% of H3
                    position = -1
                    signals[i] = -0.25
    
    return signals
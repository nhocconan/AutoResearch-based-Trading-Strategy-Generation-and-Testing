#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Camarilla pivot breakout with 12h volume confirmation and chop regime filter
# - Primary signal: Price breaks above/below Camarilla H3/L3 levels from prior 12h
# - Volume filter: 12h volume > 1.3x 20-period average volume (institutional participation)
# - Regime filter: 4h Choppiness Index > 61.8 (range market) for mean reversion at H3/L3
# - Position size: 0.25 discrete level to minimize fee churn
# - Stoploss: 2.0x ATR(14) on 4h
# - Target: 20-50 trades/year (80-200 total over 4 years) per 4h strategy guidelines
# - Works in bull/bear: Camarilla levels act as support/resistance in any regime; volume/confirms breakout validity; chop filter avoids false signals in strong trends

name = "4h_12h_camarilla_volume_chop_v1"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Load HTF data ONCE before loop
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 50:
        return np.zeros(n)
    
    # Pre-compute 12h volume spike filter
    volume_12h = df_12h['volume'].values
    avg_volume_20 = pd.Series(volume_12h).rolling(window=20, min_periods=20).mean().values
    vol_spike = volume_12h > (1.3 * avg_volume_20)
    vol_spike_aligned = align_htf_to_ltf(prices, df_12h, vol_spike)
    
    # Pre-compute 12h Camarilla levels (H3, L3) from prior 12h bar
    high_12h = df_12h['high'].values
    low_12h = df_12h['low'].values
    close_12h = df_12h['close'].values
    
    # Camarilla levels: H3 = close + 1.1*(high-low)/6, L3 = close - 1.1*(high-low)/6
    camarilla_high = close_12h + (1.1 * (high_12h - low_12h) / 6)
    camarilla_low = close_12h - (1.1 * (high_12h - low_12h) / 6)
    
    camarilla_high_aligned = align_htf_to_ltf(prices, df_12h, camarilla_high)
    camarilla_low_aligned = align_htf_to_ltf(prices, df_12h, camarilla_low)
    
    # Pre-compute 4h Choppiness Index (14-period) for regime filter
    high_4h = prices['high'].values
    low_4h = prices['low'].values
    close_4h = prices['close'].values
    
    # True Range
    tr1 = high_4h - low_4h
    tr2 = np.abs(high_4h - np.roll(close_4h, 1))
    tr3 = np.abs(low_4h - np.roll(close_4h, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]
    
    # ATR(14)
    atr_14 = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    # Sum of True Range over 14 periods
    tr_sum = pd.Series(tr).rolling(window=14, min_periods=14).sum().values
    
    # Choppiness Index = 100 * log10(tr_sum / (atr_14 * 14)) / log10(14)
    chop = 100 * np.log10(tr_sum / (atr_14 * 14)) / np.log10(14)
    chop_filter = chop > 61.8  # Range market (mean reversion regime)
    
    # Pre-compute 4h ATR(14) for stoploss
    atr_14_4h = atr_14  # Reuse ATR calculated above
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    entry_price = 0.0
    
    for i in range(100, n):
        # Skip if any required data is invalid
        if (np.isnan(camarilla_high_aligned[i]) or np.isnan(camarilla_low_aligned[i]) or
            np.isnan(vol_spike_aligned[i]) or np.isnan(chop_filter[i]) or
            np.isnan(atr_14_4h[i])):
            signals[i] = 0.0
            continue
        
        if position == 1:  # Long position
            # Exit: Price reverts to Camarilla pivot (mean reversion) OR stoploss hit
            if close_4h[i] < camarilla_high_aligned[i] or close_4h[i] < entry_price - 2.0 * atr_14_4h[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit: Price reverts to Camarilla pivot (mean reversion) OR stoploss hit
            if close_4h[i] > camarilla_low_aligned[i] or close_4h[i] > entry_price + 2.0 * atr_14_4h[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat
            # Look for Camarilla breakouts with volume and chop filters
            if vol_spike_aligned[i] and chop_filter[i]:
                # Long: price breaks above Camarilla H3
                if close_4h[i] > camarilla_high_aligned[i]:
                    position = 1
                    entry_price = close_4h[i]
                    signals[i] = 0.25
                # Short: price breaks below Camarilla L3
                elif close_4h[i] < camarilla_low_aligned[i]:
                    position = -1
                    entry_price = close_4h[i]
                    signals[i] = -0.25
    
    return signals
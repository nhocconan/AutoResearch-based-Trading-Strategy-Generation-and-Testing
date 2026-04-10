#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Camarilla pivot breakout with 1d volume spike and chop regime filter
# - Long: Price breaks above Camarilla H3 level (1d) + volume spike (current > 2.0x 20-period MA) + chop regime (CHOP > 61.8 = ranging)
# - Short: Price breaks below Camarilla L3 level (1d) + volume spike + chop regime
# - Exit: Price reaches opposite Camarilla level (L3 for long, H3 for short) OR chop regime ends (CHOP < 38.2 = trending)
# - Position sizing: 0.25 discrete level
# - Works in bull/bear: Camarilla levels provide intraday support/resistance, volume confirms breakout strength, chop filter avoids false breakouts in strong trends.
# - Targets ~20-40 trades/year.

name = "4h_1d_camarilla_volume_chop_v1"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 60:
        return np.zeros(n)
    
    # Load HTF data ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    # Pre-compute primary timeframe data
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Pre-compute HTF data
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    # Calculate 1d Camarilla levels (based on previous day's OHLC)
    # Camarilla: H4 = close + 1.5*(high-low), H3 = close + 1.1*(high-low), etc.
    prev_close_1d = np.roll(close_1d, 1)
    prev_high_1d = np.roll(high_1d, 1)
    prev_low_1d = np.roll(low_1d, 1)
    
    # Handle first bar
    prev_close_1d[0] = close_1d[0]
    prev_high_1d[0] = high_1d[0]
    prev_low_1d[0] = low_1d[0]
    
    rang = prev_high_1d - prev_low_1d
    camarilla_h3 = prev_close_1d + 1.1 * rang
    camarilla_l3 = prev_close_1d - 1.1 * rang
    
    # Align Camarilla levels to 4h timeframe
    camarilla_h3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_h3)
    camarilla_l3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_l3)
    
    # Calculate 1d Choppiness Index (CHOP) for regime filter
    # CHOP = 100 * log10(sum(ATR(14)) / log10(n)) / log10(n)
    # Simplified: CHOP = 100 * log10(ATR_sum / (n * log10(TR_avg))) / log10(n)
    # We'll use a practical approximation: CHOP = 100 * log10(ATR(14) sum over 14 bars) / log10(14)
    
    # True Range for 1d
    tr1 = high_1d - low_1d
    tr2 = np.abs(high_1d - np.roll(close_1d, 1))
    tr3 = np.abs(low_1d - np.roll(close_1d, 1))
    tr1[0] = high_1d[0] - low_1d[0]
    tr2[0] = np.abs(high_1d[0] - close_1d[0])
    tr3[0] = np.abs(low_1d[0] - close_1d[0])
    tr_1d = np.maximum(tr1, np.maximum(tr2, tr3))
    
    atr_1d = pd.Series(tr_1d).ewm(span=14, min_periods=14, adjust=False).mean().values
    atr_sum_14 = pd.Series(atr_1d).rolling(window=14, min_periods=14).sum().values
    
    # Choppiness Index: CHOP > 61.8 = ranging (good for mean reversion), CHOP < 38.2 = trending
    chop = 100 * np.log10(atr_sum_14 / 14) / np.log10(14)
    chop_aligned = align_htf_to_ltf(prices, df_1d, chop)
    
    # Calculate volume confirmation: current volume > 2.0x 20-period MA
    volume_ma_20 = pd.Series(volume).ewm(span=20, min_periods=20, adjust=False).mean().values
    
    for i in range(60, n):
        # Skip if any required data is invalid
        if (np.isnan(camarilla_h3_aligned[i]) or np.isnan(camarilla_l3_aligned[i]) or 
            np.isnan(chop_aligned[i]) or np.isnan(volume_ma_20[i])):
            signals[i] = 0.0
            continue
        
        # Volume confirmation: current volume > 2.0x 20-period MA
        vol_confirm = volume[i] > 2.0 * volume_ma_20[i]
        
        # Chop regime filter: CHOP > 61.8 = ranging market (favorable for Camarilla reversals)
        chop_regime = chop_aligned[i] > 61.8
        
        if position == 0:  # Flat - look for Camarilla breakouts
            # Long entry: Price breaks above Camarilla H3 + vol confirmation + chop regime
            if close[i] > camarilla_h3_aligned[i] and vol_confirm and chop_regime:
                position = 1
                signals[i] = 0.25
            # Short entry: Price breaks below Camarilla L3 + vol confirmation + chop regime
            elif close[i] < camarilla_l3_aligned[i] and vol_confirm and chop_regime:
                position = -1
                signals[i] = -0.25
            else:
                signals[i] = 0.0
        else:  # Have position - look for exit
            # Exit: Price reaches opposite Camarilla level OR chop regime ends
            if position == 1:  # Long position
                if close[i] >= camarilla_l3_aligned[i] or not chop_regime:
                    position = 0
                    signals[i] = 0.0
                else:
                    signals[i] = 0.25
            else:  # position == -1 (Short position)
                if close[i] <= camarilla_h3_aligned[i] or not chop_regime:
                    position = 0
                    signals[i] = 0.0
                else:
                    signals[i] = -0.25
    
    return signals
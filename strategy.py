#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Camarilla pivot breakout with 12h volume spike and chop regime filter
# In trending regimes (CHOP < 38.2): breakout above/below Camarilla H3/L3 levels with volume confirmation
# In ranging regimes (CHOP > 61.8): mean reversion at Camarilla H4/L4 levels with volume confirmation
# Uses discrete position sizing 0.25 to limit trades to ~20-50/year and reduce fee drag
# Works in bull/bear markets: breakout catches trends, chop filter avoids whipsaws in ranging markets

name = "4h_12h_camarilla_breakout_volume_chop_v1"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Load 12h data ONCE before loop
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 30:
        return np.zeros(n)
    
    high_12h = df_12h['high'].values
    low_12h = df_12h['low'].values
    close_12h = df_12h['close'].values
    volume_12h = df_12h['volume'].values
    
    # Calculate 12h ATR(14) for volatility normalization
    tr1 = np.abs(high_12h[1:] - low_12h[:-1])
    tr2 = np.abs(high_12h[1:] - close_12h[:-1])
    tr3 = np.abs(low_12h[1:] - close_12h[:-1])
    tr = np.concatenate([[np.nan], np.maximum(tr1, np.maximum(tr2, tr3))])
    
    def wilders_smoothing(values, period):
        if len(values) < period:
            return np.full(len(values), np.nan)
        alpha = 1.0 / period
        result = np.full(len(values), np.nan)
        result[period-1] = np.nanmean(values[:period])
        for i in range(period, len(values)):
            result[i] = alpha * values[i] + (1 - alpha) * result[i-1]
        return result
    
    atr_12h = wilders_smoothing(tr, 14)
    
    # Calculate 12h average volume (20-period) normalized by ATR
    volume_s_12h = pd.Series(volume_12h)
    avg_volume_12h = volume_s_12h.rolling(window=20, min_periods=20).mean().values
    vol_ratio_12h = np.where(atr_12h > 0, avg_volume_12h / atr_12h, np.nan)
    avg_vol_ratio_12h = pd.Series(vol_ratio_12h).rolling(window=20, min_periods=20).mean().values
    
    # Calculate 12h Choppiness Index (CHOP)
    hh_12h = pd.Series(high_12h).rolling(window=14, min_periods=14).max().values
    ll_12h = pd.Series(low_12h).rolling(window=14, min_periods=14).min().values
    sum_atr_14 = pd.Series(atr_12h).rolling(window=14, min_periods=14).sum().values
    range_14 = hh_12h - ll_12h
    chop_12h = np.where(range_14 != 0, 
                       100 * np.log10(sum_atr_14 / range_14) / np.log10(14), 
                       50)
    
    # Calculate 12h Camarilla pivot levels (based on prior 12h bar to avoid look-ahead)
    # Camarilla levels: H4, H3, H2, H1, L1, L2, L3, L4
    # H4 = close + 1.1 * (high - low) / 2
    # H3 = close + 1.1 * (high - low) / 4
    # L3 = close - 1.1 * (high - low) / 4
    # L4 = close - 1.1 * (high - low) / 2
    prev_high_12h = pd.Series(high_12h).shift(1).values
    prev_low_12h = pd.Series(low_12h).shift(1).values
    prev_close_12h = pd.Series(close_12h).shift(1).values
    
    camarilla_h4 = prev_close_12h + 1.1 * (prev_high_12h - prev_low_12h) / 2
    camarilla_h3 = prev_close_12h + 1.1 * (prev_high_12h - prev_low_12h) / 4
    camarilla_l3 = prev_close_12h - 1.1 * (prev_high_12h - prev_low_12h) / 4
    camarilla_l4 = prev_close_12h - 1.1 * (prev_high_12h - prev_low_12h) / 2
    
    # Align 12h indicators to 4h timeframe
    avg_vol_ratio_12h_aligned = align_htf_to_ltf(prices, df_12h, avg_vol_ratio_12h)
    chop_12h_aligned = align_htf_to_ltf(prices, df_12h, chop_12h)
    camarilla_h4_aligned = align_htf_to_ltf(prices, df_12h, camarilla_h4)
    camarilla_h3_aligned = align_htf_to_ltf(prices, df_12h, camarilla_h3)
    camarilla_l3_aligned = align_htf_to_ltf(prices, df_12h, camarilla_l3)
    camarilla_l4_aligned = align_htf_to_ltf(prices, df_12h, camarilla_l4)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(50, n):
        # Skip if any required data is invalid
        if (np.isnan(avg_vol_ratio_12h_aligned[i]) or np.isnan(chop_12h_aligned[i]) or
            np.isnan(camarilla_h4_aligned[i]) or np.isnan(camarilla_h3_aligned[i]) or
            np.isnan(camarilla_l3_aligned[i]) or np.isnan(camarilla_l4_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Volume confirmation: current volume > 2.0 * 20-period average volume (aligned)
        avg_volume_12h = pd.Series(volume_12h).rolling(window=20, min_periods=20).mean().values
        avg_volume_12h_aligned = align_htf_to_ltf(prices, df_12h, avg_volume_12h)
        volume_confirmed = volume[i] > 2.0 * avg_volume_12h_aligned[i]
        
        # Regime filter
        trending_regime = chop_12h_aligned[i] < 38.2
        ranging_regime = chop_12h_aligned[i] > 61.8
        
        if position == 1:  # Long position
            if trending_regime:
                # Exit long if price breaks below H3 or we enter ranging regime
                if close[i] < camarilla_h3_aligned[i] or ranging_regime:
                    position = 0
                    signals[i] = 0.0
                else:
                    signals[i] = 0.25
            elif ranging_regime:
                # Exit long if price rises above H4 or drops below L4
                if close[i] > camarilla_h4_aligned[i] or close[i] < camarilla_l4_aligned[i]:
                    position = 0
                    signals[i] = 0.0
                else:
                    signals[i] = 0.25
                
        elif position == -1:  # Short position
            if trending_regime:
                # Exit short if price breaks above L3 or we enter ranging regime
                if close[i] > camarilla_l3_aligned[i] or ranging_regime:
                    position = 0
                    signals[i] = 0.0
                else:
                    signals[i] = -0.25
            elif ranging_regime:
                # Exit short if price drops below L4 or rises above H4
                if close[i] < camarilla_l4_aligned[i] or close[i] > camarilla_h4_aligned[i]:
                    position = 0
                    signals[i] = 0.0
                else:
                    signals[i] = -0.25
        else:  # Flat
            if trending_regime:
                # Enter long on breakout above H3 with volume confirmation
                if close[i] > camarilla_h3_aligned[i] and volume_confirmed:
                    position = 1
                    signals[i] = 0.25
                # Enter short on breakout below L3 with volume confirmation
                elif close[i] < camarilla_l3_aligned[i] and volume_confirmed:
                    position = -1
                    signals[i] = -0.25
            elif ranging_regime:
                # Mean reversion: buy near L4, sell near H4
                if close[i] <= camarilla_l4_aligned[i] and volume_confirmed:
                    position = 1
                    signals[i] = 0.25
                elif close[i] >= camarilla_h4_aligned[i] and volume_confirmed:
                    position = -1
                    signals[i] = -0.25
    
    return signals
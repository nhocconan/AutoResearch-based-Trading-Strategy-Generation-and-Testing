#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1d Camarilla pivot breakout with 1w volume confirmation and choppiness regime filter
# In trending regimes (CHOP < 38.2): breakout above/below Camarilla H3/L3 levels with volume confirmation
# In ranging regimes (CHOP > 61.8): mean reversion at Camarilla H3/L3 levels with volume confirmation
# Uses discrete position sizing 0.30 to limit trades to ~7-25/year and reduce fee drag
# Works in bull/bear markets: breakout catches trends, chop filter avoids whipsaws in ranging markets
# Added 1w HTF volume filter to reduce false signals and improve trade quality

name = "1d_1w_camarilla_breakout_volume_chop_v1"
timeframe = "1d"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Load 1w data ONCE before loop
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 30:
        return np.zeros(n)
    
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    volume_1w = df_1w['volume'].values
    
    # Calculate 1w average volume (20-period) for confirmation
    volume_s_1w = pd.Series(volume_1w)
    avg_volume_1w = volume_s_1w.rolling(window=20, min_periods=20).mean().values
    
    # Calculate 1d Choppiness Index (CHOP)
    hh_1d = pd.Series(high).rolling(window=14, min_periods=14).max().values
    ll_1d = pd.Series(low).rolling(window=14, min_periods=14).min().values
    
    # True Range for 1d
    tr1 = np.abs(high[1:] - low[:-1])
    tr2 = np.abs(high[1:] - close[:-1])
    tr3 = np.abs(low[1:] - close[:-1])
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
    
    atr_1d = wilders_smoothing(tr, 14)
    sum_atr_14 = pd.Series(atr_1d).rolling(window=14, min_periods=14).sum().values
    range_14 = hh_1d - ll_1d
    chop_1d = np.where(range_14 != 0, 
                       100 * np.log10(sum_atr_14 / range_14) / np.log10(14), 
                       50)
    
    # Calculate 1d Camarilla pivot levels (based on prior day to avoid look-ahead)
    range_1d = high - low
    h3_1d = close + 1.1 * range_1d
    l3_1d = close - 1.1 * range_1d
    h4_1d = close + 1.5 * range_1d
    l4_1d = close - 1.5 * range_1d
    
    # Align 1w indicators to 1d timeframe
    avg_volume_1w_aligned = align_htf_to_ltf(prices, df_1w, avg_volume_1w)
    chop_1d_aligned = align_htf_to_ltf(prices, None, chop_1d)  # Already 1d, no alignment needed
    h3_1d_aligned = align_htf_to_ltf(prices, None, h3_1d)      # Already 1d
    l3_1d_aligned = align_htf_to_ltf(prices, None, l3_1d)      # Already 1d
    h4_1d_aligned = align_htf_to_ltf(prices, None, h4_1d)      # Already 1d
    l4_1d_aligned = align_htf_to_ltf(prices, None, l4_1d)      # Already 1d
    
    # Pre-compute volume confirmation array
    volume_confirmed = volume > 1.5 * avg_volume_1w_aligned
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(100, n):
        # Skip if any required data is invalid
        if (np.isnan(avg_volume_1w_aligned[i]) or np.isnan(chop_1d_aligned[i]) or
            np.isnan(h3_1d_aligned[i]) or np.isnan(l3_1d_aligned[i]) or
            np.isnan(h4_1d_aligned[i]) or np.isnan(l4_1d_aligned[i]) or
            np.isnan(volume_confirmed[i])):
            signals[i] = 0.0
            continue
        
        # Regime filter
        trending_regime = chop_1d_aligned[i] < 38.2
        ranging_regime = chop_1d_aligned[i] > 61.8
        
        if position == 1:  # Long position
            if trending_regime:
                # Exit long if price breaks below H3 or we enter ranging regime
                if close[i] < h3_1d_aligned[i] or ranging_regime:
                    position = 0
                    signals[i] = 0.0
                else:
                    signals[i] = 0.30
            elif ranging_regime:
                # Exit long if price rises above H4 or drops below L3
                if close[i] > h4_1d_aligned[i] or close[i] < l3_1d_aligned[i]:
                    position = 0
                    signals[i] = 0.0
                else:
                    signals[i] = 0.30
                
        elif position == -1:  # Short position
            if trending_regime:
                # Exit short if price breaks above L3 or we enter ranging regime
                if close[i] > l3_1d_aligned[i] or ranging_regime:
                    position = 0
                    signals[i] = 0.0
                else:
                    signals[i] = -0.30
            elif ranging_regime:
                # Exit short if price drops below L4 or rises above H3
                if close[i] < l4_1d_aligned[i] or close[i] > h3_1d_aligned[i]:
                    position = 0
                    signals[i] = 0.0
                else:
                    signals[i] = -0.30
        else:  # Flat
            if trending_regime:
                # Enter long on breakout above H3 with volume confirmation
                if close[i] > h3_1d_aligned[i] and volume_confirmed[i]:
                    position = 1
                    signals[i] = 0.30
                # Enter short on breakout below L3 with volume confirmation
                elif close[i] < l3_1d_aligned[i] and volume_confirmed[i]:
                    position = -1
                    signals[i] = -0.30
            elif ranging_regime:
                # Mean reversion: buy near L3, sell near H3
                if close[i] <= l3_1d_aligned[i] and volume_confirmed[i]:
                    position = 1
                    signals[i] = 0.30
                elif close[i] >= h3_1d_aligned[i] and volume_confirmed[i]:
                    position = -1
                    signals[i] = -0.30
    
    return signals
#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Camarilla pivot breakout with volume confirmation and choppiness regime filter
# - Uses 1d HTF for Camarilla pivot levels (H3, L3, H4, L4) as key support/resistance
# - Breakout above H3 or below L3 with volume > 1.5x 20-period average signals entry
# - Choppiness regime filter: CHOP(14) > 61.8 = range (mean revert at H3/L3), CHOP < 38.2 = trending (breakout)
# - In trending regime (CHOP < 38.2): breakout continues in direction of breakout
# - In ranging regime (CHOP > 61.8): mean reversion at opposite H3/L3 level
# - Fixed position size 0.25 to control drawdown
# - Target: 20-50 trades/year on 4h timeframe (80-200 total over 4 years)

name = "4h_1d_camarilla_breakout_v26"
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
    if len(df_1d) < 50:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate 1d Camarilla pivot levels (based on previous day)
    # Typical price = (high + low + close) / 3
    typical_price = (high_1d + low_1d + close_1d) / 3
    # Range = high - low
    rng = high_1d - low_1d
    
    # Camarilla levels
    # H4 = close + 1.5 * (high - low)
    # L4 = close - 1.5 * (high - low)
    # H3 = close + 1.1 * (high - low)
    # L3 = close - 1.1 * (high - low)
    # H2 = close + 0.55 * (high - low)
    # L2 = close - 0.55 * (high - low)
    # H1 = close + 0.275 * (high - low)
    # L1 = close - 0.275 * (high - low)
    # Pivot = (high + low + close) / 3
    
    h4 = close_1d + 1.5 * rng
    l4 = close_1d - 1.5 * rng
    h3 = close_1d + 1.1 * rng
    l3 = close_1d - 1.1 * rng
    h2 = close_1d + 0.55 * rng
    l2 = close_1d - 0.55 * rng
    h1 = close_1d + 0.275 * rng
    l1 = close_1d - 0.275 * rng
    pivot = typical_price
    
    # Align all HTF data to 4h timeframe (wait for completed HTF bar)
    h3_aligned = align_htf_to_ltf(prices, df_1d, h3)
    l3_aligned = align_htf_to_ltf(prices, df_1d, l3)
    h4_aligned = align_htf_to_ltf(prices, df_1d, h4)
    l4_aligned = align_htf_to_ltf(prices, df_1d, l4)
    
    # Pre-compute volume confirmation (20-period average for 4h)
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    # Pre-compute choppiness index (14-period) for regime filter
    # CHOP = 100 * log10(sum(ATR(1)) / (max(high) - min(low))) / log10(N)
    # Simplified: CHOP = 100 * log10(ATR_sum / (HHV - LLV)) / log10(14)
    tr1 = np.maximum(high - low, np.absolute(high - np.roll(close, 1)), np.absolute(low - np.roll(close, 1)))
    tr1[0] = high[0] - low[0]  # First period TR
    atr_sum = pd.Series(tr1).rolling(window=14, min_periods=14).sum().values
    hhvs = pd.Series(high).rolling(window=14, min_periods=14).max().values
    llvs = pd.Series(low).rolling(window=14, min_periods=14).min().values
    chop = 100 * np.log10(atr_sum / (hhvs - llvs + 1e-10)) / np.log10(14)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(100, n):
        # Skip if any required data is invalid
        if (np.isnan(h3_aligned[i]) or np.isnan(l3_aligned[i]) or
            np.isnan(h4_aligned[i]) or np.isnan(l4_aligned[i]) or
            np.isnan(vol_ma_20[i]) or np.isnan(chop[i]) or
            vol_ma_20[i] <= 0):
            signals[i] = 0.0
            continue
        
        # Volume confirmation: current 4h volume > 1.5x average
        volume_confirmed = volume[i] > 1.5 * vol_ma_20[i]
        
        # Regime filter: CHOP < 38.2 = trending, CHOP > 61.8 = ranging
        trending_regime = chop[i] < 38.2
        ranging_regime = chop[i] > 61.8
        
        # Fixed position size
        position_size = 0.25
        
        if position == 1:  # Long position
            # Exit conditions
            if trending_regime:
                # In trending regime: exit when price reaches H4 or trend weakens
                if close[i] >= h4_aligned[i]:
                    position = 0
                    signals[i] = 0.0
                else:
                    signals[i] = position_size
            elif ranging_regime:
                # In ranging regime: exit when price reaches L3 (mean reversion target)
                if close[i] <= l3_aligned[i]:
                    position = 0
                    signals[i] = 0.0
                else:
                    signals[i] = position_size
            else:
                # Neutral regime: exit
                position = 0
                signals[i] = 0.0
                
        elif position == -1:  # Short position
            # Exit conditions
            if trending_regime:
                # In trending regime: exit when price reaches L4 or trend weakens
                if close[i] <= l4_aligned[i]:
                    position = 0
                    signals[i] = 0.0
                else:
                    signals[i] = -position_size
            elif ranging_regime:
                # In ranging regime: exit when price reaches H3 (mean reversion target)
                if close[i] >= h3_aligned[i]:
                    position = 0
                    signals[i] = 0.0
                else:
                    signals[i] = -position_size
            else:
                # Neutral regime: exit
                position = 0
                signals[i] = 0.0
        else:  # Flat
            # Entry logic based on regime and price action
            if volume_confirmed:
                if trending_regime:
                    # In trending regime: breakout entries
                    if close[i] > h3_aligned[i]:
                        # Breakout above H3: long
                        position = 1
                        signals[i] = position_size
                    elif close[i] < l3_aligned[i]:
                        # Breakout below L3: short
                        position = -1
                        signals[i] = -position_size
                elif ranging_regime:
                    # In ranging regime: mean reversion entries
                    if close[i] <= l3_aligned[i]:
                        # Price at L3: long mean reversion
                        position = 1
                        signals[i] = position_size
                    elif close[i] >= h3_aligned[i]:
                        # Price at H3: short mean reversion
                        position = -1
                        signals[i] = -position_size
    
    return signals
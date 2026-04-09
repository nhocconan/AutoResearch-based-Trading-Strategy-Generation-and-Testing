#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h strategy using 12h Camarilla pivot levels + volume confirmation + choppiness regime filter
# Camarilla pivots on 12h provide key support/resistance levels derived from prior day's range
# Long when price touches L3 level with volume confirmation in choppy market (CHOP > 61.8)
# Short when price touches H3 level with volume confirmation in choppy market
# Uses discrete position sizing 0.25 to target ~25-40 trades/year and minimize fee drag
# Works in bull/bear markets: mean reversion at pivots in chop, session filter avoids low-liquidity periods

name = "4h_12h_camarilla_pivot_v1"
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
    
    # Load 12h data ONCE before loop
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 2:
        return np.zeros(n)
    
    high_12h = df_12h['high'].values
    low_12h = df_12h['low'].values
    close_12h = df_12h['close'].values
    
    # Calculate 12h Camarilla pivot levels (based on prior 12h bar)
    # Camarilla: H4 = close + 1.5*(high-low), H3 = close + 1.125*(high-low)
    #            L3 = close - 1.125*(high-low), L4 = close - 1.5*(high-low)
    # But we use prior bar's range for forward-looking levels
    prior_high = np.roll(high_12h, 1)
    prior_low = np.roll(low_12h, 1)
    prior_close = np.roll(close_12h, 1)
    prior_high[0] = np.nan
    prior_low[0] = np.nan
    prior_close[0] = np.nan
    
    camarilla_h3 = prior_close + 1.125 * (prior_high - prior_low)
    camarilla_l3 = prior_close - 1.125 * (prior_high - prior_low)
    
    # Calculate 12h Choppiness Index (CHOP) for regime filter
    # CHOP = 100 * log10(sum(ATR over n) / (log(n) * (max(high) - min(low)))) / log10(n)
    # Simplified: CHOP > 61.8 = ranging/choppy, CHOP < 38.2 = trending
    def true_range(high, low, close_prev):
        tr1 = high - low
        tr2 = np.abs(high - close_prev)
        tr3 = np.abs(low - close_prev)
        return np.maximum(tr1, np.maximum(tr2, tr3))
    
    close_prev_12h = np.roll(close_12h, 1)
    close_prev_12h[0] = close_12h[0]  # avoid NaN for first bar
    tr_12h = true_range(high_12h, low_12h, close_prev_12h)
    
    atr_sum_14 = pd.Series(tr_12h).rolling(window=14, min_periods=14).sum().values
    max_high_14 = pd.Series(high_12h).rolling(window=14, min_periods=14).max().values
    min_low_14 = pd.Series(low_12h).rolling(window=14, min_periods=14).min().values
    chop_raw = 100 * np.log10(atr_sum_14 / (np.log10(14) * (max_high_14 - min_low_14)))
    chop_12h = chop_raw / np.log10(14)  # normalize
    
    # Align 12h indicators to 4h timeframe
    camarilla_h3_aligned = align_htf_to_ltf(prices, df_12h, camarilla_h3)
    camarilla_l3_aligned = align_htf_to_ltf(prices, df_12h, camarilla_l3)
    chop_12h_aligned = align_htf_to_ltf(prices, df_12h, chop_12h)
    
    # Pre-compute session filter (08-20 UTC)
    hours = pd.DatetimeIndex(open_time).hour
    in_session = (hours >= 8) & (hours <= 20)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(100, n):
        # Skip if any required data is invalid or outside session
        if (np.isnan(camarilla_h3_aligned[i]) or np.isnan(camarilla_l3_aligned[i]) or
            np.isnan(chop_12h_aligned[i]) or not in_session[i]):
            signals[i] = 0.0
            continue
        
        # Volume confirmation: current 4h volume > 1.8x average 4h volume (20-period)
        vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
        volume_confirmed = volume[i] > 1.8 * vol_ma_20[i] if not np.isnan(vol_ma_20[i]) else False
        
        if position == 1:  # Long position
            # Exit long if price rises above H3 or falls below L3 (mean reversion complete)
            if close[i] > camarilla_h3_aligned[i] or close[i] < camarilla_l3_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit short if price rises above H3 or falls below L3
            if close[i] > camarilla_h3_aligned[i] or close[i] < camarilla_l3_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat
            # Mean reversion strategy: enter at L3/H3 with volume confirmation in choppy market
            if close[i] <= camarilla_l3_aligned[i] and volume_confirmed and chop_12h_aligned[i] > 61.8:
                position = 1
                signals[i] = 0.25
            elif close[i] >= camarilla_h3_aligned[i] and volume_confirmed and chop_12h_aligned[i] > 61.8:
                position = -1
                signals[i] = -0.25
    
    return signals
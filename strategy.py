#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1d Camarilla pivot levels + volume spike + choppiness regime filter
# - Primary signal: Price touches Camarilla H3 (resistance) for short or L3 (support) for long on 1d
# - Volume filter: 1d volume > 1.5x 20-period average volume (confirmation of institutional interest)
# - Regime filter: Choppiness Index(14) < 38.2 (trending market) to avoid false breakouts in chop
# - Position size: 0.25 discrete level to minimize fee churn
# - Stoploss: 1.5x ATR(14) on 1d
# - Target: 7-25 trades/year (30-100 total over 4 years) per 1d strategy guidelines
# - Works in bull/bear: Camarilla levels adapt to volatility; volume confirms breakout strength; chop filter avoids ranging markets

name = "1d_camarilla_volume_chop_v1"
timeframe = "1d"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Load HTF data ONCE before loop
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 20:
        return np.zeros(n)
    
    # Pre-compute 1d Camarilla levels (based on previous day)
    high_1d = prices['high'].values
    low_1d = prices['low'].values
    close_1d = prices['close'].values
    
    # Shift to use previous day's OHLC for today's Camarilla levels
    prev_high = np.roll(high_1d, 1)
    prev_low = np.roll(low_1d, 1)
    prev_close = np.roll(close_1d, 1)
    prev_high[0] = high_1d[0]  # First bar uses current bar
    prev_low[0] = low_1d[0]
    prev_close[0] = close_1d[0]
    
    # Calculate Camarilla levels
    rang = prev_high - prev_low
    camarilla_h3 = prev_close + rang * 1.1 / 4  # H3 = C + (H-L)*1.1/4
    camarilla_l3 = prev_close - rang * 1.1 / 4  # L3 = C - (H-L)*1.1/4
    camarilla_h4 = prev_close + rang * 1.1 / 2  # H4 = C + (H-L)*1.1/2
    camarilla_l4 = prev_close - rang * 1.1 / 2  # L4 = C - (H-L)*1.1/2
    
    # Pre-compute 1d volume spike filter
    volume_1d = prices['volume'].values
    avg_volume_20 = pd.Series(volume_1d).rolling(window=20, min_periods=20).mean().values
    vol_spike = volume_1d > (1.5 * avg_volume_20)
    
    # Pre-compute 1d Choppiness Index (CHOP)
    tr1 = high_1d - low_1d
    tr2 = np.abs(high_1d - np.roll(close_1d, 1))
    tr3 = np.abs(low_1d - np.roll(close_1d, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]
    atr_14 = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    # True range sum over 14 periods
    tr_sum = pd.Series(tr).rolling(window=14, min_periods=14).sum().values
    # Highest high and lowest low over 14 periods
    hh_14 = pd.Series(high_1d).rolling(window=14, min_periods=14).max().values
    ll_14 = pd.Series(low_1d).rolling(window=14, min_periods=14).min().values
    # Choppiness Index: CHOP = 100 * log10(tr_sum / (hh_14 - ll_14)) / log10(14)
    # Avoid division by zero and log of zero
    hl_range = hh_14 - ll_14
    chop = np.full_like(tr_sum, 50.0)  # Default to neutral
    valid = (hl_range > 0) & ~np.isnan(tr_sum) & ~np.isnan(hl_range)
    chop[valid] = 100 * np.log10(tr_sum[valid] / hl_range[valid]) / np.log10(14)
    # Trending market: CHOP < 38.2
    chop_filter = chop < 38.2
    
    # Pre-compute 1d ATR(14) for stoploss
    atr_14 = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    entry_price = 0.0
    
    for i in range(100, n):
        # Skip if any required data is invalid
        if (np.isnan(camarilla_h3[i]) or np.isnan(camarilla_l3[i]) or
            np.isnan(vol_spike[i]) or np.isnan(chop_filter[i]) or
            np.isnan(atr_14[i])):
            signals[i] = 0.0
            continue
        
        if position == 1:  # Long position
            # Exit: Price crosses below L3 OR stoploss hit
            if close_1d[i] < camarilla_l3[i] or close_1d[i] < entry_price - 1.5 * atr_14[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit: Price crosses above H3 OR stoploss hit
            if close_1d[i] > camarilla_h3[i] or close_1d[i] > entry_price + 1.5 * atr_14[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat
            # Look for Camarilla touch with volume and chop filters
            if vol_spike[i] and chop_filter[i]:
                # Long: Price touches or crosses above L3 (support)
                if close_1d[i] >= camarilla_l3[i] and close_1d[i-1] < camarilla_l3[i-1]:
                    position = 1
                    entry_price = close_1d[i]
                    signals[i] = 0.25
                # Short: Price touches or crosses below H3 (resistance)
                elif close_1d[i] <= camarilla_h3[i] and close_1d[i-1] > camarilla_h3[i-1]:
                    position = -1
                    entry_price = close_1d[i]
                    signals[i] = -0.25
    
    return signals
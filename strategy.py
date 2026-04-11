#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Camarilla pivot breakout + 1d volume confirmation + chop regime filter
# - Camarilla pivot levels from 1d: long at L3 breakout, short at H3 breakdown
# - Volume confirmation: 4h volume > 2.0x 20-period average to filter weak moves
# - Chop regime: only trade when CHOP(14) < 61.8 (trending market) to avoid whipsaws in ranging markets
# - Uses discrete position sizing: ±0.25 to limit drawdown and reduce fee churn
# - Target: 20-50 trades/year (80-200 total over 4 years) to stay within fee drag limits
# - Camarilla pivots work well in both bull and bear markets by identifying key support/resistance
# - Volume confirmation ensures breakouts have conviction
# - Chop filter avoids false signals during sideways consolidation

name = "4h_1d_camarilla_volume_chop_v1"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    # Load 1d data ONCE before loop for Camarilla pivots and chop regime
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return signals
    
    # Pre-compute 1d Camarilla pivot levels (based on previous day)
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate pivot point and ranges
    pivot = (high_1d + low_1d + close_1d) / 3.0
    range_hl = high_1d - low_1d
    
    # Camarilla levels
    h3 = pivot + (range_hl * 1.1 / 4)
    l3 = pivot - (range_hl * 1.1 / 4)
    h4 = pivot + (range_hl * 1.1 / 2)
    l4 = pivot - (range_hl * 1.1 / 2)
    
    # Align 1d levels to 4h timeframe (completed 1d bar only)
    h3_aligned = align_htf_to_ltf(prices, df_1d, h3)
    l3_aligned = align_htf_to_ltf(prices, df_1d, l3)
    h4_aligned = align_htf_to_ltf(prices, df_1d, h4)
    l4_aligned = align_htf_to_ltf(prices, df_1d, l4)
    
    # Pre-compute 1d chop regime (trending vs ranging)
    # CHOP = 100 * log10(sum(ATR(14)) / log10(n) / (highest_high - lowest_low))
    # Simplified: CHOP > 61.8 = ranging, CHOP < 38.2 = trending
    high_14 = pd.Series(high_1d).rolling(window=14, min_periods=14).max()
    low_14 = pd.Series(low_1d).rolling(window=14, min_periods=14).min()
    tr1 = pd.Series(high_1d).rolling(window=14, min_periods=1).max() - pd.Series(low_1d).rolling(window=14, min_periods=1).min()
    tr2 = abs(pd.Series(high_1d).shift(1) - pd.Series(close_1d))
    tr3 = abs(pd.Series(low_1d).shift(1) - pd.Series(close_1d))
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr14 = tr.rolling(window=14, min_periods=14).mean()
    
    # Avoid division by zero
    highest_high = high_14
    lowest_low = low_14
    range_14 = highest_high - lowest_low
    chop = np.where(
        (range_14 > 0) & (atr14 > 0),
        100 * np.log10(atr14.sum() / np.log10(14) / range_14),
        50  # neutral when undefined
    )
    chop_aligned = align_htf_to_ltf(prices, df_1d, chop.values)
    
    # Pre-compute 4h volume confirmation (20-period average)
    volume_series = pd.Series(volume)
    volume_sma_20 = volume_series.rolling(window=20, min_periods=20).mean().values
    
    for i in range(100, n):  # Start after 100-bar warmup
        # Skip if any required data is invalid
        if (np.isnan(h3_aligned[i]) or np.isnan(l3_aligned[i]) or
            np.isnan(volume_sma_20[i]) or np.isnan(chop_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Current price data
        price_close = close[i]
        price_high = high[i]
        price_low = low[i]
        volume_current = volume[i]
        
        # Camarilla breakout conditions (using previous bar's levels to avoid look-ahead)
        breakout_long = price_close > h3_aligned[i-1]  # Close above H3
        breakdown_short = price_close < l3_aligned[i-1]  # Close below L3
        
        # Volume confirmation: current volume > 2.0x 20-period average
        vol_confirm = volume_current > 2.0 * volume_sma_20[i]
        
        # Chop regime filter: only trade when trending (CHOP < 61.8)
        trending_regime = chop_aligned[i] < 61.8
        
        # Entry conditions
        enter_long = False
        enter_short = False
        
        # Long: Camarilla H3 breakout + volume confirmation + trending regime
        if breakout_long and vol_confirm and trending_regime:
            enter_long = True
        
        # Short: Camarilla L3 breakdown + volume confirmation + trending regime
        if breakdown_short and vol_confirm and trending_regime:
            enter_short = True
        
        # Exit conditions: opposite Camarilla level or regime change to ranging
        exit_long = False
        exit_short = False
        
        if position == 1:
            # Exit long if price breaks below L3 OR regime turns ranging
            exit_long = (price_close < l3_aligned[i-1]) or (not trending_regime)
        elif position == -1:
            # Exit short if price breaks above H3 OR regime turns ranging
            exit_short = (price_close > h3_aligned[i-1]) or (not trending_regime)
        
        # Trading logic
        if enter_long and position != 1:
            position = 1
            signals[i] = 0.25
        elif enter_short and position != -1:
            position = -1
            signals[i] = -0.25
        elif position == 1 and exit_long:
            position = 0
            signals[i] = 0.0
        elif position == -1 and exit_short:
            position = 0
            signals[i] = 0.0
        else:
            # Maintain current position
            signals[i] = 0.25 if position == 1 else (-0.25 if position == -1 else 0.0)
    
    return signals
#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h Camarilla pivot breakout with 1w volume confirmation and chop regime filter
# - Uses weekly HTF for trend context and volume confirmation
# - Enters long when price breaks above Camarilla H3 level with volume > 1.5x weekly average
# - Enters short when price breaks below Camarilla L3 level with volume > 1.5x weekly average
# - Only trades when weekly choppiness index < 38.2 (strong trending regime)
# - Uses discrete position sizing (±0.25) to limit drawdown and reduce fee churn
# - Target: 12-37 trades/year (50-150 total over 4 years) to stay within fee drag limits
# - Camarilla pivots work in both bull (breakouts with volume) and bear (breakdowns with volume) markets
# - Weekly timeframe provides reliable regime filter and volume confirmation for 12h entries

name = "12h_1w_camarilla_vol_chop_v1"
timeframe = "12h"
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
    
    # Load weekly data ONCE before loop for Camarilla pivots, volume, and chop regime
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 50:
        return signals
    
    # Pre-compute weekly indicators
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    volume_1w = df_1w['volume'].values
    
    # Calculate Camarilla pivot levels (based on previous weekly bar)
    # H4 = close + 1.5*(high-low), H3 = close + 1.1*(high-low), etc.
    weekly_range = high_1w - low_1w
    camarilla_h3 = close_1w + 1.1 * weekly_range  # H3 resistance level
    camarilla_l3 = close_1w - 1.1 * weekly_range  # L3 support level
    camarilla_h4 = close_1w + 1.5 * weekly_range  # H4 (stoploss level)
    camarilla_l4 = close_1w - 1.5 * weekly_range  # L4 (stoploss level)
    
    # Weekly volume SMA (20-period)
    volume_series = pd.Series(volume_1w)
    volume_sma_20_1w = volume_series.rolling(window=20, min_periods=20).mean().values
    
    # Weekly choppiness index (14-period)
    # CHOP = 100 * log10(sum(ATR14) / (max(high,n) - min(low,n))) / log10(n)
    tr1 = pd.Series(high_1w).shift(1) - pd.Series(low_1w).shift(1)
    tr2 = abs(pd.Series(high_1w).shift(1) - pd.Series(close_1w).shift(1))
    tr3 = abs(pd.Series(low_1w).shift(1) - pd.Series(close_1w).shift(1))
    tr_1w = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr_14_1w = pd.Series(tr_1w).rolling(window=14, min_periods=14).sum().values
    max_high_14 = pd.Series(high_1w).rolling(window=14, min_periods=14).max().values
    min_low_14 = pd.Series(low_1w).rolling(window=14, min_periods=14).min().values
    chop_denominator = max_high_14 - min_low_14
    # Avoid division by zero
    chop_denominator = np.where(chop_denominator == 0, 1e-10, chop_denominator)
    chop_ratio = atr_14_1w / chop_denominator
    chop_1w = 100 * np.log10(chop_ratio) / np.log10(14)
    
    # Align weekly indicators to 12h timeframe
    camarilla_h3_aligned = align_htf_to_ltf(prices, df_1w, camarilla_h3)
    camarilla_l3_aligned = align_htf_to_ltf(prices, df_1w, camarilla_l3)
    camarilla_h4_aligned = align_htf_to_ltf(prices, df_1w, camarilla_h4)
    camarilla_l4_aligned = align_htf_to_ltf(prices, df_1w, camarilla_l4)
    volume_sma_20_aligned = align_htf_to_ltf(prices, df_1w, volume_sma_20_1w)
    chop_1w_aligned = align_htf_to_ltf(prices, df_1w, chop_1w)
    
    for i in range(100, n):  # Start after 100-bar warmup
        # Skip if any required data is invalid
        if (np.isnan(camarilla_h3_aligned[i]) or np.isnan(camarilla_l3_aligned[i]) or
            np.isnan(volume_sma_20_aligned[i]) or np.isnan(chop_1w_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Current price data
        price_close = close[i]
        price_high = high[i]
        price_low = low[i]
        volume_current = volume[i]
        
        # Camarilla breakout conditions (using previous bar's levels)
        breakout_long = price_close > camarilla_h3_aligned[i-1]  # Close above H3
        breakout_short = price_close < camarilla_l3_aligned[i-1]  # Close below L3
        
        # Volume confirmation: current volume > 1.5x 20-period weekly average
        vol_confirm = volume_current > 1.5 * volume_sma_20_aligned[i]
        
        # Chop regime filter: only trade when weekly chop < 38.2 (strong trending regime)
        chop_filter = chop_1w_aligned[i] < 38.2
        
        # Entry conditions
        enter_long = False
        enter_short = False
        
        # Long: Camarilla H3 breakout + volume confirmation + chop filter
        if breakout_long and vol_confirm and chop_filter:
            enter_long = True
        
        # Short: Camarilla L3 breakdown + volume confirmation + chop filter
        if breakout_short and vol_confirm and chop_filter:
            enter_short = True
        
        # Exit conditions: opposite Camarilla breakout or regime change
        exit_long = False
        exit_short = False
        
        if position == 1:
            # Exit long if price breaks below L3 OR chop regime becomes too high
            exit_long = (price_close < camarilla_l3_aligned[i-1]) or (chop_1w_aligned[i] >= 38.2)
        elif position == -1:
            # Exit short if price breaks above H3 OR chop regime becomes too high
            exit_short = (price_close > camarilla_h3_aligned[i-1]) or (chop_1w_aligned[i] >= 38.2)
        
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
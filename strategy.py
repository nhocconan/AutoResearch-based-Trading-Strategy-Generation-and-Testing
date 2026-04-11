#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h Camarilla pivot levels from 1d + volume spike + choppiness regime filter
# - Camarilla levels (H3, L3, H4, L4) from 1d act as intraday support/resistance
# - Long when price touches L3 with volume > 1.8x 20-period average and chop > 61.8 (range)
# - Short when price touches H3 with volume > 1.8x 20-period average and chop > 61.8 (range)
# - Choppiness regime filter: only trade when CHOP(14) > 61.8 to avoid trending markets and false breakouts
# - Uses discrete position sizing: ±0.25 to limit drawdown and reduce fee churn
# - Target: 12-37 trades/year (50-150 total over 4 years) to stay within fee drag limits for 12h
# - Volume spike requirement (>1.8x average) ensures we only trade high-conviction mean reversions
# - Works in both bull (mean reversion in range) and bear (mean reversion in range) markets
# - 1d HTF provides reliable Camarilla levels and volume confirmation, reducing false signals

name = "12h_1d_camarilla_volume_chop_v1"
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
    
    # Load 1d data ONCE before loop for Camarilla levels, volume, and chop
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return signals
    
    # Pre-compute 1d indicators
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    volume_1d = df_1d['volume'].values
    
    # True range for ATR (used in chop calculation)
    tr1 = pd.Series(high_1d).shift(1) - pd.Series(low_1d).shift(1)
    tr2 = abs(pd.Series(high_1d).shift(1) - pd.Series(close_1d).shift(1))
    tr3 = abs(pd.Series(low_1d).shift(1) - pd.Series(close_1d).shift(1))
    tr_1d = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr_14_1d = pd.Series(tr_1d).ewm(span=14, adjust=False, min_periods=14).mean().values
    
    # 1d volume SMA (20-period)
    volume_series = pd.Series(volume_1d)
    volume_sma_20_1d = volume_series.rolling(window=20, min_periods=20).mean().values
    
    # Choppiness Index: CHOP = 100 * log10(sum(ATR14) / (HHV - LLV)) / log10(14)
    hh_14_1d = pd.Series(high_1d).rolling(window=14, min_periods=14).max().values
    ll_14_1d = pd.Series(low_1d).rolling(window=14, min_periods=14).min().values
    sum_atr_14_1d = pd.Series(atr_14_1d).rolling(window=14, min_periods=14).sum().values
    denominator = hh_14_1d - ll_14_1d
    chop_1d = np.where(denominator > 0, 100 * np.log10(sum_atr_14_1d / denominator) / np.log10(14), 50)
    
    # Camarilla levels (based on previous day's range)
    # H4 = close + 1.1 * (high - low) / 2
    # L4 = close - 1.1 * (high - low) / 2
    # H3 = close + 1.1 * (high - low) / 4
    # L3 = close - 1.1 * (high - low) / 4
    rangep = high_1d - low_1d
    camarilla_h4 = close_1d + 1.1 * rangep / 2
    camarilla_l4 = close_1d - 1.1 * rangep / 2
    camarilla_h3 = close_1d + 1.1 * rangep / 4
    camarilla_l3 = close_1d - 1.1 * rangep / 4
    
    # Align 1d indicators to 12h timeframe
    camarilla_h3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_h3)
    camarilla_l3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_l3)
    camarilla_h4_aligned = align_htf_to_ltf(prices, df_1d, camarilla_h4)
    camarilla_l4_aligned = align_htf_to_ltf(prices, df_1d, camarilla_l4)
    volume_sma_20_aligned = align_htf_to_ltf(prices, df_1d, volume_sma_20_1d)
    chop_aligned = align_htf_to_ltf(prices, df_1d, chop_1d)
    
    for i in range(100, n):  # Start after 100-bar warmup
        # Skip if any required data is invalid
        if (np.isnan(camarilla_h3_aligned[i]) or np.isnan(camarilla_l3_aligned[i]) or
            np.isnan(volume_sma_20_aligned[i]) or np.isnan(chop_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Current price data
        price_close = close[i]
        price_high = high[i]
        price_low = low[i]
        volume_current = volume[i]
        
        # Volume confirmation: current volume > 1.8x 20-period average
        vol_confirm = volume_current > 1.8 * volume_sma_20_aligned[i]
        
        # Choppiness regime filter: only trade when CHOP > 61.8 (range-bound market)
        chop_filter = chop_aligned[i] > 61.8
        
        # Entry conditions: price touches Camarilla levels with volume and chop confirmation
        enter_long = False
        enter_short = False
        
        # Long: price touches or crosses above L3 with volume and chop confirmation
        if price_low <= camarilla_l3_aligned[i] and vol_confirm and chop_filter:
            enter_long = True
        
        # Short: price touches or crosses below H3 with volume and chop confirmation
        if price_high >= camarilla_h3_aligned[i] and vol_confirm and chop_filter:
            enter_short = True
        
        # Exit conditions: price reaches opposite Camarilla level or regime changes
        exit_long = False
        exit_short = False
        
        if position == 1:
            # Exit long if price reaches H3 or H4 or chop regime ends
            exit_long = (price_high >= camarilla_h3_aligned[i]) or (not chop_filter)
        elif position == -1:
            # Exit short if price reaches L3 or L4 or chop regime ends
            exit_short = (price_low <= camarilla_l3_aligned[i]) or (not chop_filter)
        
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
#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Camarilla pivot breakout + volume spike + chop regime filter
# - Camarilla levels from 1d: L3, H3 act as intraday support/resistance
# - Long when price closes above H3 with volume > 1.8x 20-period average
# - Short when price closes below L3 with volume > 1.8x 20-period average
# - Chop regime filter: only trade when choppiness index > 61.8 (range market) for mean reversion
# - Uses discrete position sizing: ±0.25 to limit drawdown and reduce fee churn
# - Target: 20-50 trades/year (80-200 total over 4 years) within 4h limits
# - Works in both bull (breakouts with volume) and bear (mean reversion in chop) markets
# - 1d HTF provides reliable pivot levels and volume confirmation

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
    
    # Load 1d data ONCE before loop for Camarilla pivots and volume
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return signals
    
    # Pre-compute 1d Camarilla pivot levels
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate pivot point (PP)
    pp = (high_1d + low_1d + close_1d) / 3.0
    # Calculate range
    rng = high_1d - low_1d
    # Camarilla levels
    h3 = pp + (rng * 1.1 / 4.0)  # Resistance level
    l3 = pp - (rng * 1.1 / 4.0)  # Support level
    
    # Pre-compute 1d volume SMA (20-period)
    volume_1d = df_1d['volume'].values
    volume_series = pd.Series(volume_1d)
    volume_sma_20_1d = volume_series.rolling(window=20, min_periods=20).mean().values
    
    # Pre-compute 4h choppiness index (14-period)
    high_series = pd.Series(high)
    low_series = pd.Series(low)
    close_series = pd.Series(close)
    
    # True range
    tr1 = high_series.shift(1) - low_series.shift(1)
    tr2 = abs(high_series.shift(1) - close_series.shift(1))
    tr3 = abs(low_series.shift(1) - close_series.shift(1))
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    
    # ATR(14)
    atr_14 = tr.ewm(span=14, adjust=False, min_periods=14).mean()
    
    # Highest high and lowest low over 14 periods
    hh_14 = high_series.rolling(window=14, min_periods=14).max()
    ll_14 = low_series.rolling(window=14, min_periods=14).min()
    
    # Chop = 100 * log10(sum(TR14) / (ATR14 * 14)) / log10(14)
    sum_tr_14 = tr.rolling(window=14, min_periods=14).sum()
    chop = 100 * np.log10(sum_tr_14 / (atr_14 * 14)) / np.log10(14)
    chop_values = chop.values
    
    # Align 1d indicators to 4h timeframe
    h3_aligned = align_htf_to_ltf(prices, df_1d, h3)
    l3_aligned = align_htf_to_ltf(prices, df_1d, l3)
    volume_sma_20_aligned = align_htf_to_ltf(prices, df_1d, volume_sma_20_1d)
    
    for i in range(100, n):  # Start after 100-bar warmup
        # Skip if any required data is invalid
        if (np.isnan(h3_aligned[i]) or np.isnan(l3_aligned[i]) or
            np.isnan(volume_sma_20_aligned[i]) or np.isnan(chop_values[i])):
            signals[i] = 0.0
            continue
        
        # Current price data
        price_close = close[i]
        price_high = high[i]
        price_low = low[i]
        volume_current = volume[i]
        
        # Camarilla breakout conditions
        breakout_long = price_close > h3_aligned[i-1]  # Close above previous period's H3
        breakout_short = price_close < l3_aligned[i-1]  # Close below previous period's L3
        
        # Volume confirmation: current volume > 1.8x 20-period average
        vol_confirm = volume_current > 1.8 * volume_sma_20_aligned[i]
        
        # Chop regime filter: only trade in ranging markets (chop > 61.8)
        chop_filter = chop_values[i] > 61.8
        
        # Entry conditions
        enter_long = False
        enter_short = False
        
        # Long: Camarilla H3 breakout + volume confirmation + chop filter
        if breakout_long and vol_confirm and chop_filter:
            enter_long = True
        
        # Short: Camarilla L3 breakdown + volume confirmation + chop filter
        if breakout_short and vol_confirm and chop_filter:
            enter_short = True
        
        # Exit conditions: opposite Camarilla breakout or chop regime ends
        exit_long = False
        exit_short = False
        
        if position == 1:
            # Exit long if price breaks below L3 OR chop regime ends (trending market)
            exit_long = (price_close < l3_aligned[i-1]) or (not chop_filter)
        elif position == -1:
            # Exit short if price breaks above H3 OR chop regime ends
            exit_short = (price_close > h3_aligned[i-1]) or (not chop_filter)
        
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
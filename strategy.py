#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h Camarilla pivot levels from 1d + volume spike + choppiness regime
# - Camarilla pivot levels (S3/S2/S1/R1/R2/R3) calculated from 1d daily range
# - Long when price breaks above R1 with volume > 1.8x 20-period average (strong conviction)
# - Short when price breaks below S1 with volume > 1.8x 20-period average
# - Choppiness regime filter: only trade when CHOP(14) < 38.2 (trending market) to avoid sideways chop
# - Uses discrete position sizing: ±0.25 to limit drawdown and reduce fee churn
# - Target: 12-37 trades/year (50-150 total over 4 years) for 12h timeframe
# - Works in both bull (breakouts with volume) and bear (breakdowns with volume) markets
# - 1d HTF provides reliable pivot levels and volume confirmation

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
    
    # Load 1d data ONCE before loop for Camarilla pivots and volume
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return signals
    
    # Pre-compute 1d Camarilla pivot levels
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate pivot point and Camarilla levels
    pivot = (high_1d + low_1d + close_1d) / 3.0
    range_1d = high_1d - low_1d
    
    # Camarilla levels: R4 = close + range * 1.1/2, R3 = close + range * 1.1/4, etc.
    # We use R1, S1, R2, S2 for breakouts
    camarilla_r1 = close_1d + range_1d * 1.1 / 12
    camarilla_s1 = close_1d - range_1d * 1.1 / 12
    camarilla_r2 = close_1d + range_1d * 1.1 / 6
    camarilla_s2 = close_1d - range_1d * 1.1 / 6
    
    # 1d volume SMA (20-period)
    volume_1d = df_1d['volume'].values
    volume_series = pd.Series(volume_1d)
    volume_sma_20_1d = volume_series.rolling(window=20, min_periods=20).mean().values
    
    # Align 1d indicators to 12h timeframe
    camarilla_r1_aligned = align_htf_to_ltf(prices, df_1d, camarilla_r1)
    camarilla_s1_aligned = align_htf_to_ltf(prices, df_1d, camarilla_s1)
    camarilla_r2_aligned = align_htf_to_ltf(prices, df_1d, camarilla_r2)
    camarilla_s2_aligned = align_htf_to_ltf(prices, df_1d, camarilla_s2)
    volume_sma_20_aligned = align_htf_to_ltf(prices, df_1d, volume_sma_20_1d)
    
    # Pre-compute 12h Choppiness Index (CHOP)
    # CHOP = 100 * log10(sum(ATR(14)) / (log10(n) * (max(high) - min(low))))
    high_series = pd.Series(high)
    low_series = pd.Series(low)
    close_series = pd.Series(close)
    
    # True range for ATR
    tr1 = high_series.shift(1) - low_series.shift(1)
    tr2 = abs(high_series.shift(1) - close_series.shift(1))
    tr3 = abs(low_series.shift(1) - close_series.shift(1))
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr_14 = tr.rolling(window=14, min_periods=14).sum().values
    
    # Max high and min low over last 14 periods
    max_high_14 = high_series.rolling(window=14, min_periods=14).max().values
    min_low_14 = low_series.rolling(window=14, min_periods=14).min().values
    
    # Chopiness Index: CHOP = 100 * log10(ATR_sum / (log10(n) * (max_high - min_low)))
    # Simplified: CHOP = 100 * log10(sum(TR14) / (log10(14) * (HH14 - LL14)))
    log10_14 = np.log10(14)
    chop_numerator = atr_14
    chop_denominator = log10_14 * (max_high_14 - min_low_14)
    chop = 100 * np.log10(np.where(chop_denominator > 0, chop_numerator / chop_denominator, 1))
    
    for i in range(100, n):  # Start after 100-bar warmup
        # Skip if any required data is invalid
        if (np.isnan(camarilla_r1_aligned[i]) or np.isnan(camarilla_s1_aligned[i]) or
            np.isnan(volume_sma_20_aligned[i]) or np.isnan(chop[i])):
            signals[i] = 0.0
            continue
        
        # Current price data
        price_close = close[i]
        price_high = high[i]
        price_low = low[i]
        volume_current = volume[i]
        
        # Camarilla breakout conditions
        breakout_long = price_close > camarilla_r1_aligned[i-1]  # Close above previous period's R1
        breakout_short = price_close < camarilla_s1_aligned[i-1]  # Close below previous period's S1
        
        # Volume confirmation: current volume > 1.8x 20-period average (using 1d aligned volume)
        vol_confirm = volume_current > 1.8 * volume_sma_20_aligned[i]
        
        # Choppiness regime filter: only trade when CHOP < 38.2 (trending market)
        chop_filter = chop[i] < 38.2
        
        # Entry conditions
        enter_long = False
        enter_short = False
        
        # Long: Camarilla R1 breakout + volume confirmation + chop filter
        if breakout_long and vol_confirm and chop_filter:
            enter_long = True
        
        # Short: Camarilla S1 breakdown + volume confirmation + chop filter
        if breakout_short and vol_confirm and chop_filter:
            enter_short = True
        
        # Exit conditions: opposite Camarilla level break or chop regime change
        exit_long = False
        exit_short = False
        
        if position == 1:
            # Exit long if price breaks below S1 OR chop regime becomes choppy
            exit_long = (price_close < camarilla_s1_aligned[i-1]) or (chop[i] >= 38.2)
        elif position == -1:
            # Exit short if price breaks above R1 OR chop regime becomes choppy
            exit_short = (price_close > camarilla_r1_aligned[i-1]) or (chop[i] >= 38.2)
        
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
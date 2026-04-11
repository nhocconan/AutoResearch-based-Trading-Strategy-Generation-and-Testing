#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h Camarilla pivot levels from 1d + volume spike + choppiness regime filter
# - Long when price touches or breaks above Camarilla H3 level with volume > 1.8x 20-period average
# - Short when price touches or breaks below Camarilla L3 level with volume > 1.8x 20-period average
# - Choppiness regime filter: only trade when CHOP(14) < 61.8 (trending market) to avoid false signals in ranging markets
# - Uses discrete position sizing: ±0.25 to limit drawdown and reduce fee churn
# - Target: 12-37 trades/year (50-150 total over 4 years) to stay within fee drag limits for 12h
# - Volume spike requirement ensures we only trade high-conviction breakouts from pivot levels
# - Works in both bull (breakouts with volume) and bear (breakdowns with volume) markets
# - 1d HTF provides reliable Camarilla levels and volume confirmation, reducing false signals

name = "12h_1d_camarilla_volume_chop_v2"
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
    
    # Pre-compute 1d Camarilla levels (based on previous day's high, low, close)
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate Camarilla levels for each day
    camarilla_h3 = np.full_like(close_1d, np.nan)
    camarilla_l3 = np.full_like(close_1d, np.nan)
    
    for i in range(len(df_1d)):
        if i == 0:
            continue  # Skip first day (no previous day)
        prev_high = high_1d[i-1]
        prev_low = low_1d[i-1]
        prev_close = close_1d[i-1]
        range_val = prev_high - prev_low
        
        camarilla_h3[i] = prev_close + range_val * 1.1 / 4
        camarilla_l3[i] = prev_close - range_val * 1.1 / 4
    
    # Pre-compute 1d volume SMA (20-period)
    volume_1d = df_1d['volume'].values
    volume_series = pd.Series(volume_1d)
    volume_sma_20_1d = volume_series.rolling(window=20, min_periods=20).mean().values
    
    # Pre-compute 1d Choppiness Index (CHOP)
    # CHOP = 100 * log10(sum(TR over n) / (n * (max(high) - min(low)))) / log10(n)
    tr_list = []
    for i in range(len(df_1d)):
        if i == 0:
            tr_list.append(high_1d[i] - low_1d[i])  # First TR
        else:
            tr = max(
                high_1d[i] - low_1d[i],
                abs(high_1d[i] - close_1d[i-1]),
                abs(low_1d[i] - close_1d[i-1])
            )
            tr_list.append(tr)
    
    tr_series = pd.Series(tr_list)
    atr_sum = tr_series.rolling(window=14, min_periods=14).sum()
    max_high = pd.Series(high_1d).rolling(window=14, min_periods=14).max()
    min_low = pd.Series(low_1d).rolling(window=14, min_periods=14).min()
    chop_raw = 100 * np.log10(atr_sum / (14 * (max_high - min_low))) / np.log10(14)
    chop_values = chop_raw.values
    
    # Align 1d indicators to 12h timeframe
    camarilla_h3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_h3)
    camarilla_l3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_l3)
    volume_sma_20_aligned = align_htf_to_ltf(prices, df_1d, volume_sma_20_1d)
    chop_aligned = align_htf_to_ltf(prices, df_1d, chop_values)
    
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
        
        # Camarilla level touch/breakout conditions
        touch_long = price_high >= camarilla_h3_aligned[i]  # High touches or breaks above H3
        touch_short = price_low <= camarilla_l3_aligned[i]  # Low touches or breaks below L3
        
        # Volume confirmation: current volume > 1.8x 20-period average (using 1d aligned volume)
        vol_confirm = volume_current > 1.8 * volume_sma_20_aligned[i]
        
        # Choppiness regime filter: only trade when CHOP < 61.8 (trending market)
        chop_filter = chop_aligned[i] < 61.8
        
        # Entry conditions
        enter_long = False
        enter_short = False
        
        # Long: Camarilla H3 touch/breakout + volume confirmation + chop filter
        if touch_long and vol_confirm and chop_filter:
            enter_long = True
        
        # Short: Camarilla L3 touch/breakout + volume confirmation + chop filter
        if touch_short and vol_confirm and chop_filter:
            enter_short = True
        
        # Exit conditions: opposite Camarilla level touch or chop regime shift to ranging
        exit_long = False
        exit_short = False
        
        if position == 1:
            # Exit long if price touches L3 OR chop shifts to ranging (CHOP >= 61.8)
            exit_long = (price_low <= camarilla_l3_aligned[i]) or (not chop_filter)
        elif position == -1:
            # Exit short if price touches H3 OR chop shifts to ranging (CHOP >= 61.8)
            exit_short = (price_high >= camarilla_h3_aligned[i]) or (not chop_filter)
        
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
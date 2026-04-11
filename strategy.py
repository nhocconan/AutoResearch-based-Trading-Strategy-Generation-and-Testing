#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h Camarilla pivot levels from 1d + volume spike + chop regime filter
# - Camarilla levels (L3, L4, H3, H4) calculated from 1d OHLC act as intraday support/resistance
# - Long when price touches L3/L4 with volume > 1.8x 24-period 12h volume average (mean reversion in range)
# - Short when price touches H3/H4 with volume > 1.8x 24-period 12h volume average
# - Chop regime filter: only trade when Chopiness Index(14) > 61.8 (range-bound market) to avoid trending whipsaws
# - Uses discrete position sizing: ±0.25 to limit drawdown and reduce fee churn
# - Target: 12-37 trades/year (50-150 total over 4 years) to stay within fee drag limits for 12h
# - Volume spike requirement ensures we only trade high-conviction mean reversion attempts
# - Chop filter avoids false signals during strong trends, improving performance in bear markets like 2025
# - Works in both bull (mean reversion in range) and bear (range-bound bounces) markets
# - 1d HTF provides reliable Camarilla levels, reducing noise from lower timeframe

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
    
    # Load 1d data ONCE before loop for Camarilla levels and Chop filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return signals
    
    # Pre-compute 1d Camarilla levels (based on previous day's OHLC)
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate Camarilla levels for each 1d bar (using previous bar's OHLC)
    # L3 = C - (H-L)*1.12/6, L4 = C - (H-L)*1.1/2
    # H3 = C + (H-L)*1.12/6, H4 = C + (H-L)*1.1/2
    camarilla_l3 = close_1d - (high_1d - low_1d) * 1.12 / 6
    camarilla_l4 = close_1d - (high_1d - low_1d) * 1.1 / 2
    camarilla_h3 = close_1d + (high_1d - low_1d) * 1.12 / 6
    camarilla_h4 = close_1d + (high_1d - low_1d) * 1.1 / 2
    
    # Pre-compute 1d Chopiness Index (EWM-based for simplicity, using 14-period)
    # Chop = 100 * log10(sum(ATR(1)) / (ATR(14) * 14)) / log10(14)
    # Simplified: use True Range and compare short-term vs long-term average
    tr1 = pd.Series(high_1d).shift(1) - pd.Series(low_1d).shift(1)
    tr2 = abs(pd.Series(high_1d).shift(1) - pd.Series(close_1d).shift(1))
    tr3 = abs(pd.Series(low_1d).shift(1) - pd.Series(close_1d).shift(1))
    tr_1d = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr_1_1d = pd.Series(tr_1d).ewm(span=1, adjust=False).mean().values
    atr_14_1d = pd.Series(tr_1d).ewm(span=14, adjust=False, min_periods=14).mean().values
    # Chop calculation: values > 61.8 indicate ranging market
    chop_1d = 100 * np.log10(atr_1_1d * 14 / (atr_14_1d * np.log2(14))) / np.log10(14)
    chop_1d = np.where(np.isnan(chop_1d), 50, chop_1d)  # fill NaN with neutral value
    
    # Pre-compute 12h volume SMA (24-period for 12h timeframe)
    volume_12h = df_1d['volume'].values  # Note: using 1d volume as proxy for 12h (aligned later)
    volume_series = pd.Series(volume_12h)
    volume_sma_24_12h = volume_series.rolling(window=24, min_periods=24).mean().values
    
    # Align 1d indicators to 12h timeframe
    camarilla_l3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_l3)
    camarilla_l4_aligned = align_htf_to_ltf(prices, df_1d, camarilla_l4)
    camarilla_h3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_h3)
    camarilla_h4_aligned = align_htf_to_ltf(prices, df_1d, camarilla_h4)
    chop_aligned = align_htf_to_ltf(prices, df_1d, chop_1d)
    volume_sma_24_aligned = align_htf_to_ltf(prices, df_1d, volume_sma_24_12h)
    
    for i in range(100, n):  # Start after 100-bar warmup
        # Skip if any required data is invalid
        if (np.isnan(camarilla_l3_aligned[i]) or np.isnan(camarilla_l4_aligned[i]) or
            np.isnan(camarilla_h3_aligned[i]) or np.isnan(camarilla_h4_aligned[i]) or
            np.isnan(chop_aligned[i]) or np.isnan(volume_sma_24_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Current price data
        price_close = close[i]
        price_high = high[i]
        price_low = low[i]
        volume_current = volume[i]
        
        # Volume confirmation: current volume > 1.8x 24-period average (using aligned 12h volume)
        vol_confirm = volume_current > 1.8 * volume_sma_24_aligned[i]
        
        # Chop regime filter: only trade when Chop > 61.8 (range-bound market)
        chop_filter = chop_aligned[i] > 61.8
        
        # Entry conditions: price touches Camarilla levels with volume confirmation in chop regime
        touch_l3 = price_low <= camarilla_l3_aligned[i]  # Low touches L3
        touch_l4 = price_low <= camarilla_l4_aligned[i]  # Low touches L4
        touch_h3 = price_high >= camarilla_h3_aligned[i]  # High touches H3
        touch_h4 = price_high >= camarilla_h4_aligned[i]  # High touches H4
        
        enter_long = False
        enter_short = False
        
        # Long: price touches L3 or L4 + volume confirmation + chop filter
        if (touch_l3 or touch_l4) and vol_confirm and chop_filter:
            enter_long = True
        
        # Short: price touches H3 or H4 + volume confirmation + chop filter
        if (touch_h3 or touch_h4) and vol_confirm and chop_filter:
            enter_short = True
        
        # Exit conditions: price moves back toward mean (close to VWAP or midpoint) or regime change
        exit_long = False
        exit_short = False
        
        if position == 1:
            # Exit long if price moves back above midpoint of L3-L4 or chop regime ends
            midpoint_l = (camarilla_l3_aligned[i] + camarilla_l4_aligned[i]) / 2
            exit_long = (price_close > midpoint_l) or (not chop_filter)
        elif position == -1:
            # Exit short if price moves back below midpoint of H3-H4 or chop regime ends
            midpoint_h = (camarilla_h3_aligned[i] + camarilla_h4_aligned[i]) / 2
            exit_short = (price_close < midpoint_h) or (not chop_filter)
        
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
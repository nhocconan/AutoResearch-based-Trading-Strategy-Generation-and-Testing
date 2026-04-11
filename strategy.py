#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Camarilla pivot + volume spike + choppiness regime filter
# - Uses 12h Camarilla levels from prior day as support/resistance zones
# - Long when price touches S3 level with volume > 1.5x 20-period average
# - Short when price touches R3 level with volume > 1.5x 20-period average
# - Choppiness regime filter: only trade when CHOP(14) < 38.2 (trending) OR > 61.8 (range) to avoid whipsaws
# - Uses discrete position sizing: ±0.25 to limit drawdown and reduce fee churn
# - Target: 19-50 trades/year (75-200 total over 4 years) to stay within fee drag limits for 4h
# - Works in both bull (breakouts from pivot levels) and bear (reversals at pivot levels) markets
# - 12h HTF provides stable pivot levels less prone to noise than lower timeframes

name = "4h_12h_camarilla_volume_chop_v1"
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
    
    # Load 12h data ONCE before loop for Camarilla levels and volume
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 20:
        return signals
    
    # Calculate prior 12h bar's OHLC for Camarilla levels
    # Shift by 1 to use only completed 12h bars
    high_12h = df_12h['high'].values
    low_12h = df_12h['low'].values
    close_12h = df_12h['close'].values
    
    # Calculate Camarilla levels for each completed 12h bar
    # H4 = Close + 1.1*(High-Low)*1.1/2
    # L4 = Close - 1.1*(High-Low)*1.1/2
    # R3 = Close + 1.1*(High-Low)*1.1/4
    # S3 = Close - 1.1*(High-Low)*1.1/4
    # We'll use R3/S3 as entry levels
    hl_range = high_12h - low_12h
    r3_level = close_12h + 1.1 * hl_range * 1.1 / 4
    s3_level = close_12h - 1.1 * hl_range * 1.1 / 4
    
    # Align Camarilla levels to 4h timeframe
    r3_aligned = align_htf_to_ltf(prices, df_12h, r3_level)
    s3_aligned = align_htf_to_ltf(prices, df_12h, s3_level)
    
    # 12h volume SMA (20-period)
    volume_12h = df_12h['volume'].values
    volume_series = pd.Series(volume_12h)
    volume_sma_20_12h = volume_series.rolling(window=20, min_periods=20).mean().values
    volume_sma_20_aligned = align_htf_to_ltf(prices, df_12h, volume_sma_20_12h)
    
    # Calculate 4h Choppiness Index (14-period)
    # CHOP = 100 * log10(sum(TR(14)) / (ATR(14)*14)) / log10(14)
    tr1 = pd.Series(high).shift(1) - pd.Series(low).shift(1)
    tr2 = abs(pd.Series(high).shift(1) - pd.Series(close).shift(1))
    tr3 = abs(pd.Series(low).shift(1) - pd.Series(close).shift(1))
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    
    atr_14 = pd.Series(tr).ewm(span=14, adjust=False, min_periods=14).mean().values
    tr_sum_14 = pd.Series(tr).rolling(window=14, min_periods=14).sum().values
    chop = 100 * np.log10(tr_sum_14 / (atr_14 * 14)) / np.log10(14)
    
    for i in range(100, n):  # Start after 100-bar warmup
        # Skip if any required data is invalid
        if (np.isnan(r3_aligned[i]) or np.isnan(s3_aligned[i]) or
            np.isnan(volume_sma_20_aligned[i]) or np.isnan(chop[i])):
            signals[i] = 0.0
            continue
        
        # Current price data
        price_close = close[i]
        price_high = high[i]
        price_low = low[i]
        volume_current = volume[i]
        
        # Price touching Camarilla levels (with small buffer)
        touch_s3 = price_low <= s3_aligned[i] * 1.001  # Allow 0.1% buffer
        touch_r3 = price_high >= r3_aligned[i] * 0.999  # Allow 0.1% buffer
        
        # Volume confirmation: current volume > 1.5x 20-period average
        vol_confirm = volume_current > 1.5 * volume_sma_20_aligned[i]
        
        # Choppiness regime filter: avoid choppy markets (38.2 <= CHOP <= 61.8)
        chop_value = chop[i]
        chop_filter = (chop_value < 38.2) or (chop_value > 61.8)
        
        # Entry conditions
        enter_long = False
        enter_short = False
        
        # Long: Touch S3 + volume confirmation + trending/range regime
        if touch_s3 and vol_confirm and chop_filter:
            enter_long = True
        
        # Short: Touch R3 + volume confirmation + trending/range regime
        if touch_r3 and vol_confirm and chop_filter:
            enter_short = True
        
        # Exit conditions: opposite touch or volatility collapse
        exit_long = False
        exit_short = False
        
        if position == 1:
            # Exit long if price touches R3 or enters choppy zone
            exit_long = touch_r3 or not chop_filter
        elif position == -1:
            # Exit short if price touches S3 or enters choppy zone
            exit_short = touch_s3 or not chop_filter
        
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
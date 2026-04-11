#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "12h_1w_camarilla_breakout_v1"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    # Load daily and weekly data ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    df_1w = get_htf_data(prices, '1w')
    if len(df_1d) < 10 or len(df_1w) < 10:
        return signals
    
    # Calculate weekly ATR for volatility filter
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    
    # True Range for weekly
    tr1 = high_1w - low_1w
    tr2 = np.abs(high_1w - np.roll(close_1w, 1))
    tr3 = np.abs(low_1w - np.roll(close_1w, 1))
    tr1[0] = high_1w[0] - low_1w[0]
    tr2[0] = high_1w[0] - close_1w[0]
    tr3[0] = low_1w[0] - close_1w[0]
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    atr_1w = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    # Calculate daily Camarilla levels (based on previous day's OHLC)
    # Camarilla levels for the current day are calculated from previous day's data
    prev_close = np.roll(close_1d, 1)
    prev_high = np.roll(high_1d, 1)
    prev_low = np.roll(low_1d, 1)
    prev_close[0] = close_1d[0]
    prev_high[0] = high_1d[0]
    prev_low[0] = low_1d[0]
    
    # Camarilla formula: Range = (high - low)
    range_1d = prev_high - prev_low
    # Levels: H4 = close + range * 1.1/2, H3 = close + range * 1.1/4, etc.
    camarilla_h4 = prev_close + range_1d * 1.1 / 2
    camarilla_l4 = prev_close - range_1d * 1.1 / 2
    camarilla_h3 = prev_close + range_1d * 1.1 / 4
    camarilla_l3 = prev_close - range_1d * 1.1 / 4
    camarilla_h2 = prev_close + range_1d * 1.1 / 6
    camarilla_l2 = prev_close - range_1d * 1.1 / 6
    camarilla_h1 = prev_close + range_1d * 1.1 / 12
    camarilla_l1 = prev_close - range_1d * 1.1 / 12
    
    # Align to 12h timeframe
    atr_1w_aligned = align_htf_to_ltf(prices, df_1w, atr_1w)
    camarilla_h4_aligned = align_htf_to_ltf(prices, df_1d, camarilla_h4)
    camarilla_l4_aligned = align_htf_to_ltf(prices, df_1d, camarilla_l4)
    camarilla_h3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_h3)
    camarilla_l3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_l3)
    camarilla_h2_aligned = align_htf_to_ltf(prices, df_1d, camarilla_h2)
    camarilla_l2_aligned = align_htf_to_ltf(prices, df_1d, camarilla_l2)
    camarilla_h1_aligned = align_htf_to_ltf(prices, df_1d, camarilla_h1)
    camarilla_l1_aligned = align_htf_to_ltf(prices, df_1d, camarilla_l1)
    
    # Volume confirmation: 12h volume > 1.3x 20-period average
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    for i in range(100, n):
        # Skip if any required data is invalid
        if (np.isnan(atr_1w_aligned[i]) or np.isnan(camarilla_h4_aligned[i]) or 
            np.isnan(camarilla_l4_aligned[i]) or np.isnan(vol_ma_20[i])):
            signals[i] = 0.0
            continue
        
        price_close = close[i]
        volume_current = volume[i]
        
        # Volume confirmation
        vol_confirm = volume_current > 1.3 * vol_ma_20[i]
        
        # Camarilla levels for current bar
        h4 = camarilla_h4_aligned[i]
        l4 = camarilla_l4_aligned[i]
        h3 = camarilla_h3_aligned[i]
        l3 = camarilla_l3_aligned[i]
        h2 = camarilla_h2_aligned[i]
        l2 = camarilla_l2_aligned[i]
        h1 = camarilla_h1_aligned[i]
        l1 = camarilla_l1_aligned[i]
        
        # Entry conditions
        enter_long = False
        enter_short = False
        
        # Long: Price breaks above H3 with volume confirmation
        if price_close > h3 and vol_confirm:
            enter_long = True
        
        # Short: Price breaks below L3 with volume confirmation
        if price_close < l3 and vol_confirm:
            enter_short = True
        
        # Exit conditions: price returns to the opposite level
        exit_long = price_close < l3
        exit_short = price_close > h3
        
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

# Hypothesis: Camarilla breakout strategy on 12h timeframe using weekly ATR for volatility filter
# and daily Camarilla levels (H3/L3) for entry/exit. Volume confirmation ensures participation.
# Works in both bull and bear markets by capturing breakouts from key support/resistance levels.
# Target: 20-50 trades per year on 12h timeframe to avoid fee drag. Position size 0.25 limits drawdown.
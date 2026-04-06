#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h Camarilla pivot reversal with volume confirmation and weekly EMA50 trend filter
# Long when price touches L3 support + volume spike + close > weekly EMA50
# Short when price touches H3 resistance + volume spike + close < weekly EMA50
# Camarilla levels derived from prior 1d range, effective for mean reversion in ranging markets
# Weekly EMA50 filter avoids counter-trend trades in strong trends
# Target: 50-150 total trades over 4 years with controlled risk
# ATR-based stoploss to limit drawdown

name = "12h_camarilla_reversal_1d_vol_1w_ema50_v1"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Price data
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # 1d data for Camarilla pivot calculation (using prior day's range)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    
    # Calculate Camarilla levels for each 1d bar
    # Ranges: H3 = close + 1.1*(high-low)/2, L3 = close - 1.1*(high-low)/2
    range_1d = high_1d - low_1d
    camarilla_h3 = close_1d + 1.1 * range_1d / 2
    camarilla_l3 = close_1d - 1.1 * range_1d / 2
    
    # Align Camarilla levels to 12h timeframe (using prior completed 1d bar)
    camarilla_h3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_h3)
    camarilla_l3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_l3)
    
    # Weekly data for EMA50 trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 50:
        return np.zeros(n)
    
    close_1w = df_1w['close'].values
    ema50_1w = pd.Series(close_1w).ewm(span=50, min_periods=50, adjust=False).mean().values
    ema50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema50_1w)
    
    # Volume average (20-period)
    volume_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    for i in range(100, n):
        # Skip if required data not available
        if (np.isnan(camarilla_h3_aligned[i]) or np.isnan(camarilla_l3_aligned[i]) or 
            np.isnan(ema50_1w_aligned[i]) or np.isnan(volume_ma[i])):
            if position != 0:
                signals[i] = position * 0.25
            else:
                signals[i] = 0.0
            continue
        
        if position == 1:  # long position
            # Stoploss: 2 * ATR approximation using price range
            if close[i] < entry_price - 2.0 * (high[i] - low[i]):
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
            # Exit: price moves above L3 (mean reversion complete) or trend changes
            elif close[i] > camarilla_l3_aligned[i] or close[i] > ema50_1w_aligned[i]:
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
            else:
                signals[i] = 0.25
        elif position == -1:  # short position
            # Stoploss: 2 * ATR approximation
            if close[i] > entry_price + 2.0 * (high[i] - low[i]):
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
            # Exit: price moves below H3 (mean reversion complete) or trend changes
            elif close[i] < camarilla_h3_aligned[i] or close[i] < ema50_1w_aligned[i]:
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
            else:
                signals[i] = -0.25
        else:
            # Look for reversals with volume confirmation
            # Long: price touches L3 support + uptrend filter + volume spike
            if (close[i] <= camarilla_l3_aligned[i] * 1.001 and  # Allow small buffer for touch
                close[i] > ema50_1w_aligned[i] and
                volume[i] > 1.8 * volume_ma[i]):
                signals[i] = 0.25
                position = 1
                entry_price = close[i]
            # Short: price touches H3 resistance + downtrend filter + volume spike
            elif (close[i] >= camarilla_h3_aligned[i] * 0.999 and  # Allow small buffer for touch
                  close[i] < ema50_1w_aligned[i] and
                  volume[i] > 1.8 * volume_ma[i]):
                signals[i] = -0.25
                position = -1
                entry_price = close[i]
    
    return signals
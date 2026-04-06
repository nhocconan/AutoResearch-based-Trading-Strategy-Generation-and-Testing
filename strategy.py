#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6-hour Camarilla pivot levels with 1-day EMA(100) trend filter and volume confirmation (1.5x 30-period average)
# Long when price crosses above H3 level, price > 1d EMA(100), and volume > 1.5x average
# Short when price crosses below L3 level, price < 1d EMA(100), and volume > 1.5x average
# Exit when price crosses H4/L4 (strong reversal) or trend changes (price crosses EMA)
# Position size: 0.25 (25% of capital)
# Uses 1d trend to align with higher timeframe bias and reduce false signals
# Target: 100-200 total trades over 4 years (25-50/year)

name = "6h_camarilla_1d_ema_vol_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Price data
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # 1d data for EMA trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 100:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    
    # 1d EMA(100) for trend filter
    ema_1d = pd.Series(close_1d).ewm(span=100, adjust=False).mean().values
    ema_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_1d)
    
    # Calculate Camarilla levels from previous 1d bar
    # H4 = close + 1.5 * (high - low)
    # H3 = close + 1.1 * (high - low)
    # L3 = close - 1.1 * (high - low)
    # L4 = close - 1.5 * (high - low)
    # Using previous day's OHLC to avoid look-ahead
    prev_close_1d = np.roll(close_1d, 1)
    prev_high_1d = np.roll(df_1d['high'].values, 1)
    prev_low_1d = np.roll(df_1d['low'].values, 1)
    prev_close_1d[0] = close_1d[0]  # handle first bar
    prev_high_1d[0] = high[0] if len(high) > 0 else close_1d[0]
    prev_low_1d[0] = low[0] if len(low) > 0 else close_1d[0]
    
    camarilla_h3 = prev_close_1d + 1.1 * (prev_high_1d - prev_low_1d)
    camarilla_l3 = prev_close_1d - 1.1 * (prev_high_1d - prev_low_1d)
    camarilla_h4 = prev_close_1d + 1.5 * (prev_high_1d - prev_low_1d)
    camarilla_l4 = prev_close_1d - 1.5 * (prev_high_1d - prev_low_1d)
    
    # Align Camarilla levels to 6h timeframe
    h3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_h3)
    l3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_l3)
    h4_aligned = align_htf_to_ltf(prices, df_1d, camarilla_h4)
    l4_aligned = align_htf_to_ltf(prices, df_1d, camarilla_l4)
    
    # Volume average (30-period)
    volume_ma = pd.Series(volume).rolling(window=30, min_periods=30).mean().values
    
    # ATR(14) for potential stoploss (though using Camarilla/H4/L4 for exits)
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr2[0] = tr1[0]
    tr3[0] = tr1[0]
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    atr = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    for i in range(30, n):
        # Skip if required data not available
        if (np.isnan(ema_1d_aligned[i]) or np.isnan(h3_aligned[i]) or 
            np.isnan(l3_aligned[i]) or np.isnan(h4_aligned[i]) or 
            np.isnan(l4_aligned[i]) or np.isnan(volume_ma[i])):
            if position != 0:
                signals[i] = position * 0.25
            else:
                signals[i] = 0.0
            continue
        
        if position == 1:  # long position
            # Exit: price crosses below L3 (mean reversion) or crosses above H4 (strong reversal)
            # or trend turns bearish (price below EMA)
            if close[i] < l3_aligned[i] or close[i] > h4_aligned[i] or close[i] < ema_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
            else:
                signals[i] = 0.25
        elif position == -1:  # short position
            # Exit: price crosses above H3 (mean reversion) or crosses below L4 (strong reversal)
            # or trend turns bullish (price above EMA)
            if close[i] > h3_aligned[i] or close[i] < l4_aligned[i] or close[i] > ema_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
            else:
                signals[i] = -0.25
        else:
            # Look for entries with volume confirmation and trend alignment
            # Long: price crosses above H3, price above EMA (bullish trend), volume confirmation
            if (close[i] > h3_aligned[i] and
                close[i] > ema_1d_aligned[i] and
                volume[i] > 1.5 * volume_ma[i]):
                signals[i] = 0.25
                position = 1
                entry_price = close[i]
            # Short: price crosses below L3, price below EMA (bearish trend), volume confirmation
            elif (close[i] < l3_aligned[i] and
                  close[i] < ema_1d_aligned[i] and
                  volume[i] > 1.5 * volume_ma[i]):
                signals[i] = -0.25
                position = -1
                entry_price = close[i]
    
    return signals
#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6s Ichimoku cloud filter with 1w Ichimoku base line for trend direction
# Uses 1w Ichimoku base line (Kijun-sen) to establish trend direction (above/below)
# Long when: price > 1w base line, 6s price > 6s cloud top, and 6s TK cross bullish
# Short when: price < 1w base line, 6s price < 6s cloud bottom, and 6s TK cross bearish
# Exit when: price crosses opposite side of 6s cloud or 1w base line
# Stoploss: 2.5 * ATR(14)
# Position size: 0.25 (25% of capital)
# Uses weekly trend filter to avoid counter-trend trades in volatile markets
# Target: 50-150 total trades over 4 years (12-38/year)

name = "6s_ichimoku_1w_base_cloud_tk"
timeframe = "6s"
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
    
    # 1w data for Ichimoku base line (trend filter)
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 52:
        return np.zeros(n)
    
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    
    # 1w Ichimoku base line (Kijun-sen): (highest high + lowest low)/2 over 26 periods
    high_series_1w = pd.Series(high_1w)
    low_series_1w = pd.Series(low_1w)
    base_line_1w = ((high_series_1w.rolling(window=26, min_periods=26).max() + 
                     low_series_1w.rolling(window=26, min_periods=26).min()) / 2).values
    base_line_1w_aligned = align_htf_to_ltf(prices, df_1w, base_line_1w)
    
    # 6s Ichimoku components
    # Conversion line (Tenkan-sen): (9-period high + low)/2
    high_series_6s = pd.Series(high)
    low_series_6s = pd.Series(low)
    conversion_line = ((high_series_6s.rolling(window=9, min_periods=9).max() + 
                        low_series_6s.rolling(window=9, min_periods=9).min()) / 2).values
    
    # Base line (Kijun-sen): (26-period high + low)/2
    base_line_6s = ((high_series_6s.rolling(window=26, min_periods=26).max() + 
                     low_series_6s.rolling(window=26, min_periods=26).min()) / 2).values
    
    # Leading Span A (Senkou Span A): (Conversion + Base)/2 shifted 26 periods ahead
    leading_span_a = ((conversion_line + base_line_6s) / 2)
    # Leading Span B (Senkou Span B): (52-period high + low)/2 shifted 26 periods ahead
    high_52s = high_series_6s.rolling(window=52, min_periods=52).max()
    low_52s = low_series_6s.rolling(window=52, min_periods=52).min()
    leading_span_b = ((high_52s + low_52s) / 2)
    
    # Cloud top/bottom (current cloud, not shifted)
    cloud_top = np.maximum(leading_span_a, leading_span_b)
    cloud_bottom = np.minimum(leading_span_a, leading_span_b)
    
    # TK Cross: Conversion line crossing Base line
    tk_cross = np.where(conversion_line > base_line_6s, 1, 
                        np.where(conversion_line < base_line_6s, -1, 0))
    
    # ATR(14) for stoploss
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
    
    for i in range(52, n):
        # Skip if required data not available
        if (np.isnan(base_line_1w_aligned[i]) or np.isnan(cloud_top[i]) or 
            np.isnan(cloud_bottom[i]) or np.isnan(tk_cross[i]) or np.isnan(atr[i])):
            if position != 0:
                signals[i] = position * 0.25
            else:
                signals[i] = 0.0
            continue
        
        if position == 1:  # long position
            # Stoploss: 2.5 * ATR
            if close[i] < entry_price - 2.5 * atr[i]:
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
            # Exit: price below cloud bottom or below 1w base line
            elif close[i] < cloud_bottom[i] or close[i] < base_line_1w_aligned[i]:
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
            else:
                signals[i] = 0.25
        elif position == -1:  # short position
            # Stoploss: 2.5 * ATR
            if close[i] > entry_price + 2.5 * atr[i]:
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
            # Exit: price above cloud top or above 1w base line
            elif close[i] > cloud_top[i] or close[i] > base_line_1w_aligned[i]:
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
            else:
                signals[i] = -0.25
        else:
            # Look for entries with Ichimoku signals and weekly trend alignment
            # Long: price above cloud, above 1w base line, and bullish TK cross
            if (close[i] > cloud_top[i] and
                close[i] > base_line_1w_aligned[i] and
                tk_cross[i] == 1):
                signals[i] = 0.25
                position = 1
                entry_price = close[i]
            # Short: price below cloud, below 1w base line, and bearish TK cross
            elif (close[i] < cloud_bottom[i] and
                  close[i] < base_line_1w_aligned[i] and
                  tk_cross[i] == -1):
                signals[i] = -0.25
                position = -1
                entry_price = close[i]
    
    return signals
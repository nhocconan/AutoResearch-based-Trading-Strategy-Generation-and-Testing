#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h Camarilla Pivot Breakout with Volume and 1w Trend Filter
# Uses weekly EMA50 for trend direction and Camarilla pivot levels from daily for entries
# Volume confirmation ensures breakout validity. Designed for fewer trades (target 50-150/4y)
# Works in bull/bear by aligning with higher timeframe trend

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Higher timeframe data
    df_1w = get_htf_data(prices, '1w')
    df_1d = get_htf_data(prices, '1d')
    
    # Weekly EMA50 for trend
    ema_50_1w = pd.Series(df_1w['close']).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_50_1w)
    
    # Daily Camarilla pivot levels
    typical_price_1d = (df_1d['high'] + df_1d['low'] + df_1d['close']) / 3
    range_1d = df_1d['high'] - df_1d['low']
    
    # Camarilla levels
    camarilla_h4 = typical_price_1d + 1.1 * range_1d / 2
    camarilla_l4 = typical_price_1d - 1.1 * range_1d / 2
    camarilla_h3 = typical_price_1d + 1.1 * range_1d / 4
    camarilla_l3 = typical_price_1d - 1.1 * range_1d / 4
    
    # Align to lower timeframe
    camarilla_h4_aligned = align_htf_to_ltf(prices, df_1d, camarilla_h4.values)
    camarilla_l4_aligned = align_htf_to_ltf(prices, df_1d, camarilla_l4.values)
    camarilla_h3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_h3.values)
    camarilla_l3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_l3.values)
    
    # Volume confirmation (12h average)
    vol_series = pd.Series(prices['volume'])
    avg_vol = vol_series.rolling(window=20, min_periods=20).mean().shift(1).values
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    
    signals = np.zeros(n)
    position = 0
    position_size = 0.25  # 25% position
    
    start = 50  # wait for EMA warmup
    
    for i in range(start, n):
        if (np.isnan(ema_50_1w_aligned[i]) or np.isnan(camarilla_h4_aligned[i]) or
            np.isnan(camarilla_l4_aligned[i]) or np.isnan(avg_vol[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        vol = prices['volume'].iloc[i]
        
        # Trend filter: weekly EMA50 slope
        if i >= 51:
            trend_up = ema_50_1w_aligned[i] > ema_50_1w_aligned[i-1]
            trend_down = ema_50_1w_aligned[i] < ema_50_1w_aligned[i-1]
        else:
            trend_up = trend_down = False
        
        if position == 0:
            # Long: price breaks above H4 with volume, in uptrend
            if price > camarilla_h4_aligned[i] and vol > 2.0 * avg_vol[i] and trend_up:
                position = 1
                signals[i] = position_size
            # Short: price breaks below L4 with volume, in downtrend
            elif price < camarilla_l4_aligned[i] and vol > 2.0 * avg_vol[i] and trend_down:
                position = -1
                signals[i] = -position_size
            else:
                signals[i] = 0.0
        elif position == 1:
            # Exit long: price breaks below L3 or trend reverses
            if price < camarilla_l3_aligned[i] or not trend_up:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = position_size
        elif position == -1:
            # Exit short: price breaks above H3 or trend reverses
            if price > camarilla_h3_aligned[i] or not trend_down:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -position_size
    
    return signals

name = "12h_Camarilla_Pivot_WeeklyTrend_Volume"
timeframe = "12h"
leverage = 1.0
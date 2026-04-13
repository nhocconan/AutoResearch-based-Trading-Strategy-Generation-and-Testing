#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Camarilla pivot reversal with daily volume confirmation and weekly trend filter.
# Uses Camarilla pivot levels from daily data for mean reversion entries, volume confirmation to
# ensure conviction, and weekly trend filter to avoid counter-trend trades. Targets 80-150
# total trades over 4 years (20-37/year) to balance opportunity and cost. Works in both bull
# and bear markets by fading extremes in ranging markets and following trends when strong.

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Daily data for Camarilla pivots and volume
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    # Weekly data for trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 20:
        return np.zeros(n)
    
    # Calculate daily Camarilla pivot levels
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Pivot point and support/resistance levels
    pivot = (high_1d + low_1d + close_1d) / 3
    range_1d = high_1d - low_1d
    
    # Camarilla levels
    camarilla_h4 = close_1d + range_1d * 1.1 / 2
    camarilla_l4 = close_1d - range_1d * 1.1 / 2
    camarilla_h3 = close_1d + range_1d * 1.1 / 4
    camarilla_l3 = close_1d - range_1d * 1.1 / 4
    camarilla_h2 = close_1d + range_1d * 1.1 / 6
    camarilla_l2 = close_1d - range_1d * 1.1 / 6
    camarilla_h1 = close_1d + range_1d * 1.1 / 12
    camarilla_l1 = close_1d - range_1d * 1.1 / 12
    
    # Calculate weekly EMA for trend filter
    close_1w = df_1w['close'].values
    ema_20_1w = pd.Series(close_1w).ewm(span=20, adjust=False, min_periods=20).mean().values
    
    # Calculate daily volume and its 20-period average
    volume_1d = df_1d['volume'].values
    volume_ma_20_1d = pd.Series(volume_1d).rolling(window=20, min_periods=20).mean().values
    
    # Align all data to 4-hour timeframe
    camarilla_h4_aligned = align_htf_to_ltf(prices, df_1d, camarilla_h4)
    camarilla_l4_aligned = align_htf_to_ltf(prices, df_1d, camarilla_l4)
    camarilla_h3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_h3)
    camarilla_l3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_l3)
    camarilla_h2_aligned = align_htf_to_ltf(prices, df_1d, camarilla_h2)
    camarilla_l2_aligned = align_htf_to_ltf(prices, df_1d, camarilla_l2)
    camarilla_h1_aligned = align_htf_to_ltf(prices, df_1d, camarilla_h1)
    camarilla_l1_aligned = align_htf_to_ltf(prices, df_1d, camarilla_l1)
    ema_20_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_20_1w)
    volume_ma_20_1d_aligned = align_htf_to_ltf(prices, df_1d, volume_ma_20_1d)
    
    signals = np.zeros(n)
    position = 0  # -1: short, 0: flat, 1: long
    position_size = 0.25  # 25% position size
    
    for i in range(100, n):
        # Skip if any required data is not ready
        if (np.isnan(camarilla_h4_aligned[i]) or np.isnan(camarilla_l4_aligned[i]) or
            np.isnan(camarilla_h3_aligned[i]) or np.isnan(camarilla_l3_aligned[i]) or
            np.isnan(camarilla_h2_aligned[i]) or np.isnan(camarilla_l2_aligned[i]) or
            np.isnan(camarilla_h1_aligned[i]) or np.isnan(camarilla_l1_aligned[i]) or
            np.isnan(ema_20_1w_aligned[i]) or np.isnan(volume_ma_20_1d_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Volume condition: current 4h volume > 1.5x daily volume MA (adjusted for 4h)
        # 6 4h periods per day, so daily MA/6 = approximate 4h period MA
        volume_4h_approx_ma = volume_ma_20_1d_aligned[i] / 6
        volume_condition = volume[i] > (volume_4h_approx_ma * 1.5)
        
        # Trend filter: price above/below weekly EMA
        price_above_weekly_ema = close[i] > ema_20_1w_aligned[i]
        price_below_weekly_ema = close[i] < ema_20_1w_aligned[i]
        
        # Entry conditions: Camarilla reversal with volume and trend filter
        # Long when price touches L3 with volume and above weekly EMA (in uptrend)
        # Short when price touches H3 with volume and below weekly EMA (in downtrend)
        touch_l3 = low[i] <= camarilla_l3_aligned[i]
        touch_h3 = high[i] >= camarilla_h3_aligned[i]
        
        if position == 0:
            if touch_l3 and volume_condition and price_above_weekly_ema:
                position = 1
                signals[i] = position_size
            elif touch_h3 and volume_condition and price_below_weekly_ema:
                position = -1
                signals[i] = -position_size
            else:
                signals[i] = 0.0
        elif position == 1:
            # Exit when price reaches H3 or L1 (profit target or reversal)
            if high[i] >= camarilla_h3_aligned[i] or low[i] <= camarilla_l1_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = position_size
        elif position == -1:
            # Exit when price reaches L3 or H1 (profit target or reversal)
            if low[i] <= camarilla_l3_aligned[i] or high[i] >= camarilla_h1_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -position_size
    
    return signals

name = "4h_1d1w_Camarilla_Pivot_Reversal_Volume_Trend_Filter_v1"
timeframe = "4h"
leverage = 1.0
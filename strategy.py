# 1d_Camarilla_R1S1_WeeklyTrendFilter
# Hypothesis: 1d Camarilla pivot (R1/S1) breakout with weekly EMA50 trend filter and volume confirmation.
# Uses daily Camarilla pivot levels (R1, S1) derived from previous day's OHLC.
# Enters long when price breaks above R1 with volume and above weekly EMA50.
# Enters short when price breaks below S1 with volume and below weekly EMA50.
# Target: 30-100 total trades over 4 years (7-25/year). Designed for low turnover and high signal quality.
# Works in bull markets (breakout momentum) and bear markets (mean reversion via pivot rejection).
# Weekly EMA50 ensures alignment with longer-term trend, reducing false breakouts in chop.

#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1d data for Camarilla pivot calculation
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate 1d Camarilla pivot levels (R1, S1) from previous day
    # Pivot = (H + L + C) / 3
    # R1 = C + (H - L) * 1.1 / 12
    # S1 = C - (H - L) * 1.1 / 12
    typical_price = (high_1d + low_1d + close_1d) / 3
    range_hl = high_1d - low_1d
    camarilla_pivot = typical_price
    camarilla_r1 = close_1d + (range_hl * 1.1 / 12)
    camarilla_s1 = close_1d - (range_hl * 1.1 / 12)
    
    # Get weekly data for EMA50 trend filter
    df_1w = get_htf_data(prices, '1w')
    close_1w = df_1w['close'].values
    
    # Calculate weekly EMA50 for trend filter
    close_1w_series = pd.Series(close_1w)
    ema50_1w = close_1w_series.ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Align 1d and weekly indicators to 1d timeframe
    camarilla_r1_1d = align_htf_to_ltf(prices, df_1d, camarilla_r1)
    camarilla_s1_1d = align_htf_to_ltf(prices, df_1d, camarilla_s1)
    camarilla_pivot_1d = align_htf_to_ltf(prices, df_1d, camarilla_pivot)
    ema50_1w_1d = align_htf_to_ltf(prices, df_1w, ema50_1w)
    
    # Volume filter: current volume > 2.0 * 20-period average (stricter to reduce trades)
    volume_ma20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # -1: short, 0: flat, 1: long
    
    start_idx = 50  # Need sufficient data for EMA50 and volume MA
    
    for i in range(start_idx, n):
        # Skip if any required data is not available
        if (np.isnan(camarilla_r1_1d[i]) or 
            np.isnan(camarilla_s1_1d[i]) or 
            np.isnan(ema50_1w_1d[i]) or 
            np.isnan(volume_ma20[i])):
            signals[i] = 0.0
            continue
        
        # Volume filter: spike > 2.0x average (stricter to reduce trades)
        volume_filter = volume[i] > (2.0 * volume_ma20[i])
        
        # Trend filter: price above/below weekly EMA50
        price_above_ema = close[i] > ema50_1w_1d[i]
        price_below_ema = close[i] < ema50_1w_1d[i]
        
        # Price relative to Camarilla levels
        price_above_r1 = close[i] > camarilla_r1_1d[i]
        price_below_s1 = close[i] < camarilla_s1_1d[i]
        
        if position == 0:
            # Long: Price breaks above Camarilla R1 with volume and above weekly EMA50
            if (price_above_r1 and price_above_ema and volume_filter):
                signals[i] = 0.25
                position = 1
            # Short: Price breaks below Camarilla S1 with volume and below weekly EMA50
            elif (price_below_s1 and price_below_ema and volume_filter):
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit long: Price crosses below Camarilla pivot OR below weekly EMA50
            if (close[i] < camarilla_pivot_1d[i]) or (close[i] < ema50_1w_1d[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: Price crosses above Camarilla pivot OR above weekly EMA50
            if (close[i] > camarilla_pivot_1d[i]) or (close[i] > ema50_1w_1d[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "1d_Camarilla_R1S1_WeeklyTrendFilter"
timeframe = "1d"
leverage = 1.0
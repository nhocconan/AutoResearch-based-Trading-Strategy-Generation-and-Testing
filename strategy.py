#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12-hour Camarilla pivot levels from daily data with volume confirmation and 1-week trend filter
# Long when price touches Camarilla L3 level with volume >1.3x 24-period average and price above 1w EMA50
# Short when price touches Camarilla H3 level with volume >1.3x 24-period average and price below 1w EMA50
# Exit when price crosses Camarilla L4/H4 levels or reaches opposite H3/L3 level
# Uses 12-hour timeframe to capture swing trades with lower frequency to minimize fee drag
# Camarilla levels provide institutional support/resistance, volume confirms institutional interest
# Weekly EMA50 filter ensures alignment with higher timeframe trend to avoid counter-trend trades
# Target: 50-150 total trades over 4 years (12-37/year) for optimal balance of opportunity and fee efficiency

def generate_signals(prices):
    n = len(prices)
    if n < 60:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Load daily and weekly data ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    df_1w = get_htf_data(prices, '1w')
    
    # Calculate Camarilla pivot levels from daily data
    # Based on previous day's high, low, close
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Camarilla levels calculation
    # Pivot = (H + L + C) / 3
    # Range = H - L
    # L3 = C - (Range * 1.1 / 4)
    # H3 = C + (Range * 1.1 / 4)
    # L4 = C - (Range * 1.1 / 2)
    # H4 = C + (Range * 1.1 / 2)
    pivot = (high_1d + low_1d + close_1d) / 3
    rang = high_1d - low_1d
    camarilla_l3 = close_1d - (rang * 1.1 / 4)
    camarilla_h3 = close_1d + (rang * 1.1 / 4)
    camarilla_l4 = close_1d - (rang * 1.1 / 2)
    camarilla_h4 = close_1d + (rang * 1.1 / 2)
    
    # Calculate 1-week EMA50
    close_1w = df_1w['close'].values
    ema_50_1w = pd.Series(close_1w).ewm(span=50, min_periods=50, adjust=False).mean().values
    
    # Calculate volume average (24-period for 12h timeframe = 12 days)
    vol_ma = pd.Series(volume).rolling(window=24, min_periods=24).mean().values
    
    # Align indicators to 12h timeframe
    camarilla_l3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_l3)
    camarilla_h3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_h3)
    camarilla_l4_aligned = align_htf_to_ltf(prices, df_1d, camarilla_l4)
    camarilla_h4_aligned = align_htf_to_ltf(prices, df_1d, camarilla_h4)
    ema_50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_50_1w)
    vol_ma_aligned = align_htf_to_ltf(prices, df_1d, vol_ma)  # Volume MA aligned via daily
    
    signals = np.zeros(n)
    position = 0
    position_size = 0.25  # 25% position size
    
    # Start after enough data for calculations
    start = 60  # for 24-period volume MA and EMA50
    
    for i in range(start, n):
        # Skip if any critical data is NaN
        if (np.isnan(camarilla_l3_aligned[i]) or np.isnan(camarilla_h3_aligned[i]) or 
            np.isnan(camarilla_l4_aligned[i]) or np.isnan(camarilla_h4_aligned[i]) or
            np.isnan(ema_50_1w_aligned[i]) or np.isnan(vol_ma_aligned[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        vol_current = volume[i]
        
        if position == 0:
            # Long setup: price touches Camarilla L3 with volume confirmation and above weekly EMA50
            if (price <= camarilla_l3_aligned[i] * 1.001 and  # Allow small slippage
                price >= camarilla_l4_aligned[i] * 0.999 and   # But above L4 to avoid false signals
                vol_current > 1.3 * vol_ma_aligned[i] and      # Volume confirmation
                price > ema_50_1w_aligned[i]):                 # Above weekly EMA50 for bullish bias
                position = 1
                signals[i] = position_size
            # Short setup: price touches Camarilla H3 with volume confirmation and below weekly EMA50
            elif (price >= camarilla_h3_aligned[i] * 0.999 and   # Allow small slippage
                  price <= camarilla_h4_aligned[i] * 1.001 and   # But below H4 to avoid false signals
                  vol_current > 1.3 * vol_ma_aligned[i] and      # Volume confirmation
                  price < ema_50_1w_aligned[i]):                 # Below weekly EMA50 for bearish bias
                position = -1
                signals[i] = -position_size
            else:
                signals[i] = 0.0
        elif position == 1:
            # Exit long: price crosses L4 or reaches H3 level
            if (price < camarilla_l4_aligned[i] * 0.999 or 
                price > camarilla_h3_aligned[i] * 1.001):
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = position_size
        elif position == -1:
            # Exit short: price crosses H4 or reaches L3 level
            if (price > camarilla_h4_aligned[i] * 1.001 or 
                price < camarilla_l3_aligned[i] * 0.999):
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -position_size
    
    return signals

name = "12h_Camarilla_Daily_WeeklyEMA50_Volume"
timeframe = "12h"
leverage = 1.0
#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Ichimoku Cloud with 1w trend filter and volume confirmation
# - Long when price > 6h Kumo (Senkou Span A/B) AND TK Cross bullish (Tenkan > Kijun) AND 1w close > 1w EMA50 (bullish trend)
# - Short when price < 6h Kumo AND TK Cross bearish (Tenkan < Kijun) AND 1w close < 1w EMA50 (bearish trend)
# - Volume confirmation: 6h volume > 1.3x 20-period volume SMA
# - Exit: price crosses Kumo in opposite direction or loss of volume confirmation
# - Position sizing: 0.25 discrete level
# - Ichimoku parameters: Tenkan=9, Kijun=26, Senkou Span B=52, displacement=26
# - Target: 12-37 trades/year on 6h timeframe to stay within fee drag limits

name = "6h_1w_ichimoku_trend_volume_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Load HTF data ONCE before loop
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 60:
        return np.zeros(n)
    
    # Pre-compute primary timeframe data
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    # Calculate 6h Ichimoku components
    # Tenkan-sen (Conversion Line): (9-period high + 9-period low)/2
    period9_high = pd.Series(high).rolling(window=9, min_periods=9).max().values
    period9_low = pd.Series(low).rolling(window=9, min_periods=9).min().values
    tenkan = (period9_high + period9_low) / 2
    
    # Kijun-sen (Base Line): (26-period high + 26-period low)/2
    period26_high = pd.Series(high).rolling(window=26, min_periods=26).max().values
    period26_low = pd.Series(low).rolling(window=26, min_periods=26).min().values
    kijun = (period26_high + period26_low) / 2
    
    # Senkou Span A (Leading Span A): (Tenkan + Kijun)/2 shifted 26 periods ahead
    senkou_a = ((tenkan + kijun) / 2)
    
    # Senkou Span B (Leading Span B): (52-period high + 52-period low)/2 shifted 26 periods ahead
    period52_high = pd.Series(high).rolling(window=52, min_periods=52).max().values
    period52_low = pd.Series(low).rolling(window=52, min_periods=52).min().values
    senkou_b = ((period52_high + period52_low) / 2)
    
    # Calculate 1w EMA50 for trend filter
    close_1w = df_1w['close'].values
    ema_50_1w = pd.Series(close_1w).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_50_1w)
    
    # Calculate 1w close for trend comparison
    close_1w_aligned = align_htf_to_ltf(prices, df_1w, close_1w)
    
    # Calculate 6h volume SMA for regime filter
    volume_sma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    for i in range(100, n):  # Start after warmup for Ichimoku (52+26)
        # Skip if any required data is invalid
        if (np.isnan(tenkan[i]) or np.isnan(kijun[i]) or 
            np.isnan(senkou_a[i]) or np.isnan(senkou_b[i]) or
            np.isnan(ema_50_1w_aligned[i]) or np.isnan(close_1w_aligned[i]) or
            np.isnan(volume_sma_20[i])):
            signals[i] = 0.0
            continue
        
        # Kumo (Cloud) boundaries - Senkou Span A and B
        upper_kumo = np.maximum(senkou_a[i], senkou_b[i])
        lower_kumo = np.minimum(senkou_a[i], senkou_b[i])
        
        # TK Cross (Tenkan-Kijun cross)
        tk_bullish = tenkan[i] > kijun[i]
        tk_bearish = tenkan[i] < kijun[i]
        
        # Price vs Kumo
        price_above_kumo = close[i] > upper_kumo
        price_below_kumo = close[i] < lower_kumo
        
        # Volume confirmation: 6h volume > 1.3x 20-period volume SMA
        vol_confirm = volume[i] > 1.3 * volume_sma_20[i]
        
        # Trend filter: 1w close vs 1w EMA50
        trend_bullish = close_1w_aligned[i] > ema_50_1w_aligned[i]
        trend_bearish = close_1w_aligned[i] < ema_50_1w_aligned[i]
        
        # Entry conditions
        enter_long = price_above_kumo and tk_bullish and vol_confirm and trend_bullish
        enter_short = price_below_kumo and tk_bearish and vol_confirm and trend_bearish
        
        # Exit conditions: price crosses Kumo in opposite direction or loss of volume confirmation
        exit_long = close[i] < lower_kumo or not vol_confirm
        exit_short = close[i] > upper_kumo or not vol_confirm
        
        if position == 0:  # Flat - look for entry
            if enter_long:
                position = 1
                signals[i] = 0.25
            elif enter_short:
                position = -1
                signals[i] = -0.25
            else:
                signals[i] = 0.0
        elif position == 1:  # Long position - look for exit
            if exit_long:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
        else:  # position == -1 (Short position) - look for exit
            if exit_short:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
    
    return signals
#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h strategy using weekly Camarilla pivot breakout with 1d EMA34 trend filter and volume confirmation
# Long when price breaks above weekly Camarilla R4 AND 1d EMA34 > EMA55 AND volume > 2.0 * avg_volume(20)
# Short when price breaks below weekly Camarilla S4 AND 1d EMA34 < EMA55 AND volume > 2.0 * avg_volume(20)
# Exit when price crosses 1d EMA34 (trend reversal signal)
# Uses discrete sizing 0.25 to balance return and drawdown control
# Target: 75-200 total trades over 4 years (19-50/year) for 6h timeframe
# Weekly Camarilla provides strong structure with clear breakout levels from longer-term context
# 1d EMA34/EMA55 filter ensures alignment with intermediate daily trend
# Volume confirmation filters weak breakouts
# Works in bull (breakouts above R4 in uptrend) and bear (breakdowns below S4 in downtrend)

name = "6h_weeklyCamarilla_R4_S4_Breakout_1dEMA34Trend_Volume"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get weekly data ONCE before loop for Camarilla pivots
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 5:  # Need sufficient data for weekly pivots
        return np.zeros(n)
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    
    # Calculate weekly Camarilla levels (based on previous week)
    # Camarilla formula: R4 = C + ((H-L) * 1.1/2), S4 = C - ((H-L) * 1.1/2)
    # where C = (H+L+C)/3 (typical price)
    typical_price_1w = (high_1w + low_1w + close_1w) / 3.0
    camarilla_r4_1w = typical_price_1w + ((high_1w - low_1w) * 1.1 / 2.0)
    camarilla_s4_1w = typical_price_1w - ((high_1w - low_1w) * 1.1 / 2.0)
    
    # Get 1d data ONCE before loop for EMA trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 55:  # Need sufficient data for EMA55
        return np.zeros(n)
    close_1d = df_1d['close'].values
    
    # Calculate 1d EMA34 and EMA55 for trend filter
    close_series_1d = pd.Series(close_1d)
    ema_34_1d = close_series_1d.ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_55_1d = close_series_1d.ewm(span=55, adjust=False, min_periods=55).mean().values
    
    # Align HTF indicators to 6h timeframe (wait for completed HTF bar)
    camarilla_r4_aligned = align_htf_to_ltf(prices, df_1w, camarilla_r4_1w)
    camarilla_s4_aligned = align_htf_to_ltf(prices, df_1w, camarilla_s4_1w)
    ema_34_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    ema_55_aligned = align_htf_to_ltf(prices, df_1d, ema_55_1d)
    
    # Calculate volume confirmation: volume > 2.0 * 20-period average volume
    avg_volume_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_confirm = volume > (2.0 * avg_volume_20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(100, n):  # Start after warmup period
        # Skip if any value is NaN
        if (np.isnan(camarilla_r4_aligned[i]) or np.isnan(camarilla_s4_aligned[i]) or 
            np.isnan(ema_34_aligned[i]) or np.isnan(ema_55_aligned[i]) or np.isnan(avg_volume_20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: price breaks above weekly Camarilla R4 with 1d EMA34 > EMA55 and volume confirmation
            if (close[i] > camarilla_r4_aligned[i] and close[i-1] <= camarilla_r4_aligned[i-1] and 
                ema_34_aligned[i] > ema_55_aligned[i] and volume_confirm[i]):
                signals[i] = 0.25
                position = 1
            # Short: price breaks below weekly Camarilla S4 with 1d EMA34 < EMA55 and volume confirmation
            elif (close[i] < camarilla_s4_aligned[i] and close[i-1] >= camarilla_s4_aligned[i-1] and 
                  ema_34_aligned[i] < ema_55_aligned[i] and volume_confirm[i]):
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: price crosses below 1d EMA34 (trend reversal)
            if close[i] < ema_34_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: price crosses above 1d EMA34 (trend reversal)
            if close[i] > ema_34_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals
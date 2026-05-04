#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Ichimoku Cloud with 1d trend filter and volume confirmation
# Ichimoku provides dynamic support/resistance via cloud (Senkou Span A/B) and momentum via TK cross.
# 1d EMA50 ensures alignment with higher timeframe trend to avoid counter-trend entries.
# Volume confirmation (>1.5x 20 EMA) filters weak breakouts.
# Works in bull/bear: cloud acts as dynamic trend filter, TK cross catches momentum shifts.
# Target: 50-150 total trades over 4 years = 12-37/year. Discrete sizing 0.25.

name = "6h_Ichimoku_1dEMA50_VolumeConfirm"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Get 1d data for trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # Calculate 1d EMA50 for trend direction
    close_1d = pd.Series(df_1d['close'])
    ema50_1d = close_1d.ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Align 1d EMA50 to 6h timeframe (completed 1d bar only)
    ema50_aligned = align_htf_to_ltf(prices, df_1d, ema50_1d)
    
    # Calculate Ichimoku components on 6h timeframe
    # Tenkan-sen (Conversion Line): (9-period high + 9-period low)/2
    period9_high = pd.Series(high).rolling(window=9, min_periods=9).max()
    period9_low = pd.Series(low).rolling(window=9, min_periods=9).min()
    tenkan_sen = (period9_high + period9_low) / 2.0
    
    # Kijun-sen (Base Line): (26-period high + 26-period low)/2
    period26_high = pd.Series(high).rolling(window=26, min_periods=26).max()
    period26_low = pd.Series(low).rolling(window=26, min_periods=26).min()
    kijun_sen = (period26_high + period26_low) / 2.0
    
    # Senkou Span A (Leading Span A): (Tenkan-sen + Kijun-sen)/2 shifted 26 periods ahead
    senkou_a = ((tenkan_sen + kijun_sen) / 2.0)
    
    # Senkou Span B (Leading Span B): (52-period high + 52-period low)/2 shifted 26 periods ahead
    period52_high = pd.Series(high).rolling(window=52, min_periods=52).max()
    period52_low = pd.Series(low).rolling(window=52, min_periods=52).min()
    senkou_b = ((period52_high + period52_low) / 2.0)
    
    # Chikou Span (Lagging Span): Close shifted 26 periods behind (not used for entry)
    
    # For cloud top/bottom at current point, we need values shifted back 26 periods
    # Since Senkou Span A/B are plotted 26 periods ahead, we shift them back by 26 to align with current price
    senkou_a_lagged = senkou_a.shift(26).values
    senkou_b_lagged = senkou_b.shift(26).values
    
    # Volume confirmation: 20-period EMA of volume on 6h timeframe
    vol_ema_20 = pd.Series(volume).ewm(span=20, adjust=False, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(100, n):
        # Skip if any value is NaN
        if (np.isnan(ema50_aligned[i]) or np.isnan(tenkan_sen.iloc[i]) or 
            np.isnan(kijun_sen.iloc[i]) or np.isnan(senkou_a_lagged[i]) or 
            np.isnan(senkou_b_lagged[i]) or np.isnan(vol_ema_20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Volume confirmation: current volume > 1.5 x 20-period EMA
        volume_confirm = volume[i] > (1.5 * vol_ema_20[i])
        
        # Determine cloud top and bottom
        cloud_top = max(senkou_a_lagged[i], senkou_b_lagged[i])
        cloud_bottom = min(senkou_a_lagged[i], senkou_b_lagged[i])
        
        # TK cross: Tenkan-sen crossing above/below Kijun-sen
        tk_cross_up = tenkan_sen.iloc[i] > kijun_sen.iloc[i] and tenkan_sen.iloc[i-1] <= kijun_sen.iloc[i-1]
        tk_cross_down = tenkan_sen.iloc[i] < kijun_sen.iloc[i] and tenkan_sen.iloc[i-1] >= kijun_sen.iloc[i-1]
        
        if position == 0:
            # Long conditions: price above cloud + TK cross up + uptrend + volume spike
            if (close[i] > cloud_top and tk_cross_up and 
                close[i] > ema50_aligned[i] and volume_confirm):
                signals[i] = 0.25
                position = 1
            # Short conditions: price below cloud + TK cross down + downtrend + volume spike
            elif (close[i] < cloud_bottom and tk_cross_down and 
                  close[i] < ema50_aligned[i] and volume_confirm):
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: price drops below cloud OR TK cross down OR trend changes
            if (close[i] < cloud_bottom or 
                tk_cross_down or 
                close[i] < ema50_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: price rises above cloud OR TK cross up OR trend changes
            if (close[i] > cloud_top or 
                tk_cross_up or 
                close[i] > ema50_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals
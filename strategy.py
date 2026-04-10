#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Ichimoku Cloud + 1d ADX trend filter
# - Uses Ichimoku (Tenkan/Kijun/Senkou Span) on 6h for momentum and support/resistance
# - 1d ADX > 25 filters for trending markets only (avoid chop)
# - Long when price > Cloud AND Tenkan > Kijun AND 1d ADX > 25
# - Short when price < Cloud AND Tenkan < Kijun AND 1d ADX > 25
# - Exit when price crosses back into Cloud or Tenkan/Kijun cross reverses
# - Ichimoku works in both bull/bear by adapting to trend direction via cloud
# - ADX filter ensures we only trade strong trends, reducing whipsaw
# - Target: 12-30 trades/year on 6h (50-120 total over 4 years)

name = "6h_1d_ichimoku_adx_trend_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Load HTF data ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # Pre-compute Ichimoku components (9, 26, 52 periods)
    # Tenkan-sen (Conversion Line): (9-period high + 9-period low)/2
    period9_high = pd.Series(prices['high']).rolling(window=9, min_periods=9).max().values
    period9_low = pd.Series(prices['low']).rolling(window=9, min_periods=9).min().values
    tenkan = (period9_high + period9_low) / 2
    
    # Kijun-sen (Base Line): (26-period high + 26-period low)/2
    period26_high = pd.Series(prices['high']).rolling(window=26, min_periods=26).max().values
    period26_low = pd.Series(prices['low']).rolling(window=26, min_periods=26).min().values
    kijun = (period26_high + period26_low) / 2
    
    # Senkou Span A (Leading Span A): (Tenkan + Kijun)/2 shifted 26 periods ahead
    senkou_a = ((tenkan + kijun) / 2)
    
    # Senkou Span B (Leading Span B): (52-period high + 52-period low)/2 shifted 26 periods ahead
    period52_high = pd.Series(prices['high']).rolling(window=52, min_periods=52).max().values
    period52_low = pd.Series(prices['low']).rolling(window=52, min_periods=52).min().values
    senkou_b = ((period52_high + period52_low) / 2)
    
    # Pre-compute 1d ADX for trend filter
    df_1d_high = df_1d['high'].values
    df_1d_low = df_1d['low'].values
    df_1d_close = df_1d['close'].values
    
    # True Range
    tr1 = df_1d_high[1:] - df_1d_low[1:]
    tr2 = np.abs(df_1d_high[1:] - df_1d_close[:-1])
    tr3 = np.abs(df_1d_low[1:] - df_1d_close[:-1])
    tr = np.concatenate([[np.nan], np.maximum(tr1, np.maximum(tr2, tr3))])
    
    # Directional Movement
    dm_plus = np.concatenate([[np.nan], np.where((df_1d_high[1:] - df_1d_high[:-1]) > (df_1d_low[:-1] - df_1d_low[1:]), 
                                                 np.maximum(df_1d_high[1:] - df_1d_high[:-1], 0), 0)])
    dm_minus = np.concatenate([[np.nan], np.where((df_1d_low[:-1] - df_1d_low[1:]) > (df_1d_high[1:] - df_1d_high[:-1]), 
                                                  np.maximum(df_1d_low[:-1] - df_1d_low[1:], 0), 0)])
    
    # Smoothed TR, DM+, DM- (14-period)
    tr14 = pd.Series(tr).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    dm_plus_14 = pd.Series(dm_plus).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    dm_minus_14 = pd.Series(dm_minus).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    
    # DI+ and DI-
    di_plus = 100 * dm_plus_14 / tr14
    di_minus = 100 * dm_minus_14 / tr14
    
    # DX and ADX
    dx = 100 * np.abs(di_plus - di_minus) / (di_plus + di_minus)
    adx = pd.Series(dx).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    
    # Pre-compute volume confirmation: > 1.3x 20-period average
    volume_20_avg = prices['volume'].rolling(window=20, min_periods=20).mean().values
    vol_spike = prices['volume'] > (1.3 * volume_20_avg)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    # Pre-compute aligned 1d data
    adx_1d_aligned = align_htf_to_ltf(prices, df_1d, adx)
    
    for i in range(52, n):  # Start after Ichimoku warmup
        # Skip if any required data is invalid
        if (np.isnan(tenkan[i]) or np.isnan(kijun[i]) or np.isnan(senkou_a[i]) or 
            np.isnan(senkou_b[i]) or np.isnan(adx_1d_aligned[i]) or np.isnan(volume_20_avg[i])):
            # Hold current position or flat
            if position == 0:
                signals[i] = 0.0
            elif position == 1:
                signals[i] = 0.25
            else:
                signals[i] = -0.25
            continue
        
        # Determine cloud boundaries (Senkou Span A/B)
        # Note: Senkou spans are plotted 26 periods ahead, so we use current values
        upper_cloud = np.maximum(senkou_a[i], senkou_b[i])
        lower_cloud = np.minimum(senkou_a[i], senkou_b[i])
        
        if position == 0:  # Flat - look for new trend entries
            # Long when price above cloud, bullish TK cross, strong trend, volume spike
            if (prices['close'].iloc[i] > upper_cloud and 
                tenkan[i] > kijun[i] and 
                adx_1d_aligned[i] > 25 and 
                vol_spike.iloc[i]):
                position = 1
                signals[i] = 0.25
            # Short when price below cloud, bearish TK cross, strong trend, volume spike
            elif (prices['close'].iloc[i] < lower_cloud and 
                  tenkan[i] < kijun[i] and 
                  adx_1d_aligned[i] > 25 and 
                  vol_spike.iloc[i]):
                position = -1
                signals[i] = -0.25
            else:
                signals[i] = 0.0
        else:  # Have position - look for exit
            # Exit when price crosses back into cloud or TK cross reverses
            exit_signal = False
            if position == 1:  # Long position
                if (prices['close'].iloc[i] < lower_cloud or tenkan[i] < kijun[i]):
                    exit_signal = True
            elif position == -1:  # Short position
                if (prices['close'].iloc[i] > upper_cloud or tenkan[i] > kijun[i]):
                    exit_signal = True
            
            if exit_signal:
                position = 0
                signals[i] = 0.0
            else:
                if position == 1:
                    signals[i] = 0.25
                else:
                    signals[i] = -0.25
    
    return signals
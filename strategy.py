#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Ichimoku Cloud breakout with 1d trend filter and volume confirmation
# Uses Ichimoku TK cross and cloud color from daily timeframe for trend direction.
# Enters on 6h price breaking above/below the daily cloud with volume spike.
# Works in bull markets (trend following) and bear markets (counter-trend at cloud edges).
# Target: 12-37 trades/year (50-150 over 4 years) with discrete sizing to minimize fees.

name = "6h_Ichimoku_Cloud_Breakout_1dTrend_Volume"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    open_time = prices['open_time']
    
    # Session filter: 08-20 UTC (pre-compute to avoid datetime64 issues)
    hours = pd.DatetimeIndex(open_time).hour
    in_session = (hours >= 8) & (hours <= 20)
    
    # Get 1d data for Ichimoku calculation
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 52:  # Need 26*2 for Ichimoku
        return np.zeros(n)
    
    # Calculate Ichimoku components on 1d (using previous day's data to avoid look-ahead)
    # Tenkan-sen (Conversion Line): (9-period high + 9-period low)/2
    high_9 = pd.Series(df_1d['high'].values).rolling(window=9, min_periods=9).max().shift(1)
    low_9 = pd.Series(df_1d['low'].values).rolling(window=9, min_periods=9).min().shift(1)
    tenkan = ((high_9 + low_9) / 2).values
    
    # Kijun-sen (Base Line): (26-period high + 26-period low)/2
    high_26 = pd.Series(df_1d['high'].values).rolling(window=26, min_periods=26).max().shift(1)
    low_26 = pd.Series(df_1d['low'].values).rolling(window=26, min_periods=26).min().shift(1)
    kijun = ((high_26 + low_26) / 2).values
    
    # Senkou Span A (Leading Span A): (Tenkan + Kijun)/2 plotted 26 periods ahead
    senkou_a = ((tenkan + kijun) / 2)
    
    # Senkou Span B (Leading Span B): (52-period high + 52-period low)/2 plotted 26 periods ahead
    high_52 = pd.Series(df_1d['high'].values).rolling(window=52, min_periods=52).max().shift(1)
    low_52 = pd.Series(df_1d['low'].values).rolling(window=52, min_periods=52).min().shift(1)
    senkou_b = ((high_52 + low_52) / 2)
    
    # Current cloud boundaries (Senkou Span A and B from 26 periods ago)
    senkou_a_current = np.roll(senkou_a, 26)
    senkou_b_current = np.roll(senkou_b, 26)
    # First 26 values will be invalid due to roll, but we align properly via align_htf_to_ltf
    
    # Cloud color: green if Senkou A > Senkou B (bullish), red otherwise
    cloud_green = senkou_a_current > senkou_b_current
    
    # Price relative to cloud
    price_above_cloud = (df_1d['close'].values > np.maximum(senkou_a_current, senkou_b_current))
    price_below_cloud = (df_1d['close'].values < np.minimum(senkou_a_current, senkou_b_current))
    
    # TK Cross: Tenkan crosses above/below Kijun
    tk_cross_up = (tenkan > kijun) & (np.roll(tenkan, 1) <= np.roll(kijun, 1))
    tk_cross_down = (tenkan < kijun) & (np.roll(tenkan, 1) >= np.roll(kijun, 1))
    
    # 1d volume spike (volume > 2.0 * 20-period EMA of volume)
    vol_ema_20 = pd.Series(df_1d['volume'].values).ewm(span=20, adjust=False, min_periods=20).mean().values
    volume_spike = df_1d['volume'].values > (2.0 * vol_ema_20)
    
    # Align all 1d indicators to 6h timeframe
    tenkan_aligned = align_htf_to_ltf(prices, df_1d, tenkan)
    kijun_aligned = align_htf_to_ltf(prices, df_1d, kijun)
    senkou_a_aligned = align_htf_to_ltf(prices, df_1d, senkou_a_current)
    senkou_b_aligned = align_htf_to_ltf(prices, df_1d, senkou_b_current)
    cloud_green_aligned = align_htf_to_ltf(prices, df_1d, cloud_green.astype(float))
    price_above_cloud_aligned = align_htf_to_ltf(prices, df_1d, price_above_cloud.astype(float))
    price_below_cloud_aligned = align_htf_to_ltf(prices, df_1d, price_below_cloud.astype(float))
    tk_cross_up_aligned = align_htf_to_ltf(prices, df_1d, tk_cross_up.astype(float))
    tk_cross_down_aligned = align_htf_to_ltf(prices, df_1d, tk_cross_down.astype(float))
    volume_spike_aligned = align_htf_to_ltf(prices, df_1d, volume_spike)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(52, n):
        # Skip if any value is NaN or outside session
        if (np.isnan(tenkan_aligned[i]) or np.isnan(kijun_aligned[i]) or 
            np.isnan(senkou_a_aligned[i]) or np.isnan(senkou_b_aligned[i]) or 
            np.isnan(volume_spike_aligned[i]) or not in_session[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Determine if price is above/below cloud
        above_cloud = close[i] > max(senkou_a_aligned[i], senkou_b_aligned[i])
        below_cloud = close[i] < min(senkou_a_aligned[i], senkou_b_aligned[i])
        
        # Determine cloud color (bullish/bearish)
        bullish_cloud = senkou_a_aligned[i] > senkou_b_aligned[i]
        
        if position == 0:
            # Long: Price breaks above cloud with TK cross up and volume spike in bullish cloud
            # OR strong breakout above cloud regardless of TK cross
            if ((above_cloud and tk_cross_up_aligned[i] and bullish_cloud and volume_spike_aligned[i]) or
                (close[i] > senkou_a_aligned[i] and close[i] > senkou_b_aligned[i] and volume_spike_aligned[i] and
                 close[i] > max(senkou_a_aligned[i-1], senkou_b_aligned[i-1]) * 1.02)):  # 2% breakout confirmation
                signals[i] = 0.25
                position = 1
            # Short: Price breaks below cloud with TK cross down and volume spike in bearish cloud
            # OR strong breakdown below cloud regardless of TK cross
            elif ((below_cloud and tk_cross_down_aligned[i] and not bullish_cloud and volume_spike_aligned[i]) or
                  (close[i] < senkou_a_aligned[i] and close[i] < senkou_b_aligned[i] and volume_spike_aligned[i] and
                   close[i] < min(senkou_a_aligned[i-1], senkou_b_aligned[i-1]) * 0.98)):  # 2% breakdown confirmation
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: Price breaks below cloud or TK cross down in bearish cloud
            if below_cloud or (tk_cross_down_aligned[i] and not bullish_cloud):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: Price breaks above cloud or TK cross up in bullish cloud
            if above_cloud or (tk_cross_up_aligned[i] and bullish_cloud):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals
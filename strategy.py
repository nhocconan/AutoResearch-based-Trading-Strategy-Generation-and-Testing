#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Ichimoku Cloud with 1d trend filter and volume confirmation
# Long when Tenkan > Kijun AND price > Cloud AND 1d EMA50 uptrend AND volume > 1.5x 20-bar avg
# Short when Tenkan < Kijun AND price < Cloud AND 1d EMA50 downtrend AND volume > 1.5x 20-bar avg
# Exit when Tenkan/Kijun cross reverses OR price crosses opposite Cloud edge
# Uses Ichimoku for trend/momentum, 1d EMA50 for higher-timeframe trend filter, volume for confirmation
# Works in bull markets (trend continuation via cloud breakouts) and bear markets (mean reversion via cloud rejections)
# Target: 50-150 total trades over 4 years (12-37/year) on 6h.

name = "6h_Ichimoku_Cloud_1dEMA50_VolumeConfirm_v1"
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
    
    # Get 6h data for Ichimoku calculation
    # Tenkan-sen (Conversion Line): (9-period high + 9-period low) / 2
    period_tenkan = 9
    highest_9 = pd.Series(high).rolling(window=period_tenkan, min_periods=period_tenkan).max().values
    lowest_9 = pd.Series(low).rolling(window=period_tenkan, min_periods=period_tenkan).min().values
    tenkan = (highest_9 + lowest_9) / 2
    
    # Kijun-sen (Base Line): (26-period high + 26-period low) / 2
    period_kijun = 26
    highest_26 = pd.Series(high).rolling(window=period_kijun, min_periods=period_kijun).max().values
    lowest_26 = pd.Series(low).rolling(window=period_kijun, min_periods=period_kijun).min().values
    kijun = (highest_26 + lowest_26) / 2
    
    # Senkou Span A (Leading Span A): (Tenkan + Kijun) / 2
    senkou_a = (tenkan + kijun) / 2
    
    # Senkou Span B (Leading Span B): (52-period high + 52-period low) / 2
    period_senkou_b = 52
    highest_52 = pd.Series(high).rolling(window=period_senkou_b, min_periods=period_senkou_b).max().values
    lowest_52 = pd.Series(low).rolling(window=period_senkou_b, min_periods=period_senkou_b).min().values
    senkou_b = (highest_52 + lowest_52) / 2
    
    # Get 1d data for EMA50 trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    # Calculate EMA(50) on 1d data
    ema_50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    # Align EMA50 to 6h timeframe
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # Volume confirmation: >1.5x 20-bar average volume
    volume_series = pd.Series(volume)
    volume_ma_20 = volume_series.rolling(window=20, min_periods=20).mean().values
    volume_confirm = volume > 1.5 * volume_ma_20
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(52, 20)  # Senkou B needs 52 periods, volume MA needs 20
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(tenkan[i]) or np.isnan(kijun[i]) or np.isnan(senkou_a[i]) or 
            np.isnan(senkou_b[i]) or np.isnan(ema_50_1d_aligned[i]) or np.isnan(volume_ma_20[i])):
            signals[i] = 0.0
            continue
        
        vol_conf = volume_confirm[i]
        curr_close = close[i]
        tenkan_val = tenkan[i]
        kijun_val = kijun[i]
        senkou_a_val = senkou_a[i]
        senkou_b_val = senkou_b[i]
        ema_50 = ema_50_1d_aligned[i]
        
        # Determine cloud top and bottom
        cloud_top = max(senkou_a_val, senkou_b_val)
        cloud_bottom = min(senkou_a_val, senkou_b_val)
        
        # Determine 1d trend
        is_uptrend = ema_50 > close_1d[-1] if len(close_1d) > 0 else False  # Simplified trend check
        is_downtrend = ema_50 < close_1d[-1] if len(close_1d) > 0 else False
        
        # Handle exits and position management
        if position == 1:  # Long position
            # Exit: Tenkan/Kijun cross down OR price falls below cloud bottom
            if (tenkan_val < kijun_val) or (curr_close < cloud_bottom):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit: Tenkan/Kijun cross up OR price rises above cloud top
            if (tenkan_val > kijun_val) or (curr_close > cloud_top):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
                
        else:  # Flat - look for new entries
            # Long when Tenkan > Kijun AND price > cloud AND 1d EMA50 uptrend AND volume confirmation
            if (tenkan_val > kijun_val) and (curr_close > cloud_top) and (ema_50 > close[i]) and vol_conf:
                signals[i] = 0.25
                position = 1
            # Short when Tenkan < Kijun AND price < cloud AND 1d EMA50 downtrend AND volume confirmation
            elif (tenkan_val < kijun_val) and (curr_close < cloud_bottom) and (ema_50 < close[i]) and vol_conf:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
    
    return signals
#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Ichimoku Cloud with 1d trend filter and volume confirmation
# Long when price is above 1d Ichimoku cloud, Tenkan/Kijun cross bullish on 6h, volume > 1.5x average
# Short when price is below 1d Ichimoku cloud, Tenkan/Kijun cross bearish on 6h, volume > 1.5x average
# Exit when price re-enters the 1d Ichimoku cloud
# Uses discrete position sizing (0.25) and volume filter to target 12-37 trades/year.
# Ichimoku cloud acts as dynamic support/resistance that adapts to volatility, working in both bull and bear markets.

name = "6h_Ichimoku_1dTrend_VolumeConfirm_v1"
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
    open_time = prices['open_time'].values
    
    # Precompute session filter (08-20 UTC)
    hours = pd.DatetimeIndex(open_time).hour
    in_session = (hours >= 8) & (hours <= 20)
    
    # Get 1d data for Ichimoku cloud (primary trend filter)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 52:  # Need 52 for Senkou B (26*2)
        return np.zeros(n)
    
    # Calculate 1d Ichimoku components
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Tenkan-sen (Conversion Line): (9-period high + 9-period low)/2
    period9_high = pd.Series(high_1d).rolling(window=9, min_periods=9).max().values
    period9_low = pd.Series(low_1d).rolling(window=9, min_periods=9).min().values
    tenkan_sen = (period9_high + period9_low) / 2.0
    
    # Kijun-sen (Base Line): (26-period high + 26-period low)/2
    period26_high = pd.Series(high_1d).rolling(window=26, min_periods=26).max().values
    period26_low = pd.Series(low_1d).rolling(window=26, min_periods=26).min().values
    kijun_sen = (period26_high + period26_low) / 2.0
    
    # Senkou Span A (Leading Span A): (Tenkan-sen + Kijun-sen)/2 shifted 26 periods ahead
    senkou_a = ((tenkan_sen + kijun_sen) / 2.0)
    
    # Senkou Span B (Leading Span B): (52-period high + 52-period low)/2 shifted 26 periods ahead
    period52_high = pd.Series(high_1d).rolling(window=52, min_periods=52).max().values
    period52_low = pd.Series(low_1d).rolling(window=52, min_periods=52).min().values
    senkou_b = ((period52_high + period52_low) / 2.0)
    
    # Align 1d Ichimoku to 6h timeframe (wait for completed 1d bar + 26-period shift for cloud)
    tenkan_aligned = align_htf_to_ltf(prices, df_1d, tenkan_sen)
    kijun_aligned = align_htf_to_ltf(prices, df_1d, kijun_sen)
    senkou_a_aligned = align_htf_to_ltf(prices, df_1d, senkou_a, additional_delay_bars=26)
    senkou_b_aligned = align_htf_to_ltf(prices, df_1d, senkou_b, additional_delay_bars=26)
    
    # Get 6h data for Tenkan/Kijun cross (entry timing)
    df_6h = get_htf_data(prices, '6h')
    if len(df_6h) < 9:
        return np.zeros(n)
    
    # Calculate 6h Tenkan-sen and Kijun-sen for cross
    high_6h = df_6h['high'].values
    low_6h = df_6h['low'].values
    
    period9_high_6h = pd.Series(high_6h).rolling(window=9, min_periods=9).max().values
    period9_low_6h = pd.Series(low_6h).rolling(window=9, min_periods=9).min().values
    tenkan_6h = (period9_high_6h + period9_low_6h) / 2.0
    
    period26_high_6h = pd.Series(high_6h).rolling(window=26, min_periods=26).max().values
    period26_low_6h = pd.Series(low_6h).rolling(window=26, min_periods=26).min().values
    kijun_6h = (period26_high_6h + period26_low_6h) / 2.0
    
    # Align 6h indicators (no additional delay needed for cross)
    tenkan_6h_aligned = align_htf_to_ltf(prices, df_6h, tenkan_6h)
    kijun_6h_aligned = align_htf_to_ltf(prices, df_6h, kijun_6h)
    
    # Calculate 20-period average volume for confirmation
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(52, 26, 20)  # Warmup for Ichimoku components and volume
    
    for i in range(start_idx, n):
        # Skip if not in trading session
        if not in_session[i]:
            signals[i] = 0.0
            continue
            
        # Skip if any required data is NaN
        if (np.isnan(tenkan_aligned[i]) or np.isnan(kijun_aligned[i]) or 
            np.isnan(senkou_a_aligned[i]) or np.isnan(senkou_b_aligned[i]) or
            np.isnan(tenkan_6h_aligned[i]) or np.isnan(kijun_6h_aligned[i]) or
            np.isnan(vol_ma_20[i])):
            signals[i] = 0.0
            continue
        
        curr_close = close[i]
        curr_high = high[i]
        curr_low = low[i]
        curr_volume = volume[i]
        curr_tenkan = tenkan_aligned[i]
        curr_kijun = kijun_aligned[i]
        curr_senkou_a = senkou_a_aligned[i]
        curr_senkou_b = senkou_b_aligned[i]
        curr_tenkan_6h = tenkan_6h_aligned[i]
        curr_kijun_6h = kijun_6h_aligned[i]
        curr_vol_ma = vol_ma_20[i]
        
        # Determine cloud boundaries (Senkou A and B)
        cloud_top = max(curr_senkou_a, curr_senkou_b)
        cloud_bottom = min(curr_senkou_a, curr_senkou_b)
        
        # Handle exits and position management
        if position == 1:  # Long position
            # Exit: price re-enters the cloud (below cloud top)
            if curr_close < cloud_top:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit: price re-enters the cloud (above cloud bottom)
            if curr_close > cloud_bottom:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
                
        else:  # Flat - look for new entries
            # Volume confirmation: current volume > 1.5x 20-period average
            vol_confirmed = curr_volume > 1.5 * curr_vol_ma
            
            # Bullish TK cross: Tenkan crosses above Kijun on 6h
            tk_bullish = curr_tenkan_6h > curr_kijun_6h
            tk_bullish_prev = tenkan_6h_aligned[i-1] <= kijun_6h_aligned[i-1] if i > 0 else False
            
            # Bearish TK cross: Tenkan crosses below Kijun on 6h
            tk_bearish = curr_tenkan_6h < curr_kijun_6h
            tk_bearish_prev = tenkan_6h_aligned[i-1] >= kijun_6h_aligned[i-1] if i > 0 else False
            
            # Long when price above cloud, bullish TK cross on 6h, volume confirmed
            if curr_close > cloud_top and tk_bullish and tk_bullish_prev and vol_confirmed:
                signals[i] = 0.25
                position = 1
            # Short when price below cloud, bearish TK cross on 6h, volume confirmed
            elif curr_close < cloud_bottom and tk_bearish and tk_bearish_prev and vol_confirmed:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
    
    return signals
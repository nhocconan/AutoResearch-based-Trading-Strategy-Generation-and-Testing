#!/usr/bin/env python3
"""
6h_Ichimoku_TK_Cross_1dCloud_Filter_VolumeSpike
Hypothesis: On 6h timeframe, use Ichimoku Tenkan/Kijun cross with 1d cloud filter (price above/below cloud) and volume spike confirmation. 
The cloud acts as dynamic support/resistance and trend filter, reducing false signals in sideways markets. 
Volume spike ensures breakouts have conviction. Designed for 12-25 trades/year on BTC/ETH with controlled risk.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Precompute session hours (08-20 UTC) once before loop
    hours = pd.DatetimeIndex(prices["open_time"]).hour
    in_session = (hours >= 8) & (hours <= 20)
    
    # 1d data for Ichimoku cloud (loaded ONCE)
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Ichimoku components: Tenkan-sen (9-period), Kijun-sen (26-period), Senkou Span A/B (52-period)
    # Tenkan-sen = (Highest High + Lowest Low) / 2 over 9 periods
    highest_high_9 = pd.Series(high_1d).rolling(window=9, min_periods=9).max().values
    lowest_low_9 = pd.Series(low_1d).rolling(window=9, min_periods=9).min().values
    tenkan = (highest_high_9 + lowest_low_9) / 2
    
    # Kijun-sen = (Highest High + Lowest Low) / 2 over 26 periods
    highest_high_26 = pd.Series(high_1d).rolling(window=26, min_periods=26).max().values
    lowest_low_26 = pd.Series(low_1d).rolling(window=26, min_periods=26).min().values
    kijun = (highest_high_26 + lowest_low_26) / 2
    
    # Senkou Span A = (Tenkan + Kijun) / 2 shifted 26 periods ahead
    senkou_a = ((tenkan + kijun) / 2)
    
    # Senkou Span B = (Highest High + Lowest Low) / 2 over 52 periods, shifted 26 periods ahead
    highest_high_52 = pd.Series(high_1d).rolling(window=52, min_periods=52).max().values
    lowest_low_52 = pd.Series(low_1d).rolling(window=52, min_periods=52).min().values
    senkou_b = ((highest_high_52 + lowest_low_52) / 2)
    
    # The cloud is between Senkou Span A and Senkou Span B
    # For simplicity, we use the average of Senkou A and B as cloud midpoint, but actually need to check if price is above/both or below/both
    # We'll define: price above cloud if price > max(senkou_a, senkou_b)
    # price below cloud if price < min(senkou_a, senkou_b)
    senkou_a_shifted = np.roll(senkou_a, 26)  # shift ahead by 26 periods
    senkou_b_shifted = np.roll(senkou_b, 26)  # shift ahead by 26 periods
    # Fill the first 26 values with nan (since shifted)
    senkou_a_shifted[:26] = np.nan
    senkou_b_shifted[:26] = np.nan
    
    # Align Ichimoku components to 6h timeframe
    tenkan_aligned = align_htf_to_ltf(prices, df_1d, tenkan)
    kijun_aligned = align_htf_to_ltf(prices, df_1d, kijun)
    senkou_a_aligned = align_htf_to_ltf(prices, df_1d, senkou_a_shifted)
    senkou_b_aligned = align_htf_to_ltf(prices, df_1d, senkou_b_shifted)
    
    # Volume confirmation: current volume > 2.0 * 20-period average (strong spike)
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_confirm = volume > (vol_ma * 2.0)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    # Start index: need enough for Ichimoku calculations (52 for Senkou B)
    start_idx = 52
    
    for i in range(start_idx, n):
        # Skip if not in trading session
        if not in_session[i]:
            signals[i] = 0.0
            continue
        
        # Skip if any data not ready
        if (np.isnan(tenkan_aligned[i]) or np.isnan(kijun_aligned[i]) or 
            np.isnan(senkou_a_aligned[i]) or np.isnan(senkou_b_aligned[i]) or 
            np.isnan(vol_ma[i])):
            signals[i] = 0.0
            continue
        
        curr_close = close[i]
        curr_high = high[i]
        curr_low = low[i]
        
        # Ichimoku signals: Tenkan/Kijun cross
        tk_cross_up = tenkan_aligned[i] > kijun_aligned[i]
        tk_cross_down = tenkan_aligned[i] < kijun_aligned[i]
        
        # Cloud filter: price above cloud (bullish) or below cloud (bearish)
        cloud_top = np.maximum(senkou_a_aligned[i], senkou_b_aligned[i])
        cloud_bottom = np.minimum(senkou_a_aligned[i], senkou_b_aligned[i])
        price_above_cloud = curr_close > cloud_top
        price_below_cloud = curr_close < cloud_bottom
        
        if position == 0:
            # Look for entry signals with TK cross, cloud filter, and volume confirmation
            # Long: Tenkan crosses above Kijun AND price above cloud AND volume confirmation
            long_signal = tk_cross_up and price_above_cloud and volume_confirm[i]
            # Short: Tenkan crosses below Kijun AND price below cloud AND volume confirmation
            short_signal = tk_cross_down and price_below_cloud and volume_confirm[i]
            
            if long_signal:
                signals[i] = 0.25
                position = 1
                entry_price = curr_close
            elif short_signal:
                signals[i] = -0.25
                position = -1
                entry_price = curr_close
            else:
                signals[i] = 0.0
        elif position == 1:
            # Long position: exit conditions
            # Stoploss: 2.5 * ATR below entry (using 6h ATR)
            tr1 = high[1:] - low[1:]
            tr2 = np.abs(high[1:] - close[:-1])
            tr3 = np.abs(low[1:] - close[:-1])
            tr = np.concatenate([[np.nan], np.maximum(tr1, np.maximum(tr2, tr3))])
            atr = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
            
            if curr_close < entry_price - 2.5 * atr[i]:
                signals[i] = 0.0
                position = 0
            # Exit if Tenkan crosses below Kijun (trend change) or price goes below cloud
            elif tk_cross_down or not price_above_cloud:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short position: exit conditions
            # Calculate 6h ATR (same as above)
            tr1 = high[1:] - low[1:]
            tr2 = np.abs(high[1:] - close[:-1])
            tr3 = np.abs(low[1:] - close[:-1])
            tr = np.concatenate([[np.nan], np.maximum(tr1, np.maximum(tr2, tr3))])
            atr = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
            
            if curr_close > entry_price + 2.5 * atr[i]:
                signals[i] = 0.0
                position = 0
            # Exit if Tenkan crosses above Kijun (trend change) or price goes above cloud
            elif tk_cross_up or not price_below_cloud:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "6h_Ichimoku_TK_Cross_1dCloud_Filter_VolumeSpike"
timeframe = "6h"
leverage = 1.0
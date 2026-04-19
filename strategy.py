# 6h_1wVWAP_1dIchimoku_Trend_Filter
# Uses weekly VWAP as long-term value anchor and daily Ichimoku cloud for trend direction.
# Long when price is above weekly VWAP and above daily Ichimoku cloud (bullish alignment).
# Short when price is below weekly VWAP and below daily Ichimoku cloud (bearish alignment).
# Target: 20-40 trades/year per symbol. Works in bull (buy above cloud/VWAP) and bear (sell below cloud/VWAP).
# Weekly VWAP filters out short-term noise, daily Ichimoku provides clear trend context.
# Both indicators are lagging and use only historical data, ensuring no look-ahead.

#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "6h_1wVWAP_1dIchimoku_Trend_Filter"
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
    
    # 1-week data for VWAP calculation
    df_1w = get_htf_data(prices, '1w')
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    volume_1w = df_1w['volume'].values
    
    # Calculate VWAP: cumulative (price * volume) / cumulative volume
    # Typical price for VWAP: (high + low + close) / 3
    typical_price_1w = (high_1w + low_1w + close_1w) / 3.0
    pv_1w = typical_price_1w * volume_1w
    cum_pv_1w = np.cumsum(pv_1w)
    cum_vol_1w = np.cumsum(volume_1w)
    # Avoid division by zero
    vwap_1w = np.divide(cum_pv_1w, cum_vol_1w, out=np.full_like(cum_pv_1w, np.nan), where=cum_vol_1w!=0)
    
    # 1-day data for Ichimoku components
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Ichimoku parameters: 9, 26, 52
    # Tenkan-sen (Conversion Line): (9-period high + low) / 2
    high_9 = pd.Series(high_1d).rolling(window=9, min_periods=9).max()
    low_9 = pd.Series(low_1d).rolling(window=9, min_periods=9).min()
    tenkan_sen = (high_9 + low_9) / 2.0
    
    # Kijun-sen (Base Line): (26-period high + low) / 2
    high_26 = pd.Series(high_1d).rolling(window=26, min_periods=26).max()
    low_26 = pd.Series(low_1d).rolling(window=26, min_periods=26).min()
    kijun_sen = (high_26 + low_26) / 2.0
    
    # Senkou Span A (Leading Span A): (Tenkan-sen + Kijun-sen) / 2
    senkou_span_a = (tenkan_sen + kijun_sen) / 2.0
    
    # Senkou Span B (Leading Span B): (52-period high + low) / 2
    high_52 = pd.Series(high_1d).rolling(window=52, min_periods=52).max()
    low_52 = pd.Series(low_1d).rolling(window=52, min_periods=52).min()
    senkou_span_b = (high_52 + low_52) / 2.0
    
    # Align weekly VWAP to 6h timeframe
    vwap_1w_aligned = align_htf_to_ltf(prices, df_1w, vwap_1w)
    
    # Align daily Ichimoku components to 6h timeframe
    tenkan_sen_aligned = align_htf_to_ltf(prices, df_1d, tenkan_sen.values)
    kijun_sen_aligned = align_htf_to_ltf(prices, df_1d, kijun_sen.values)
    senkou_span_a_aligned = align_htf_to_ltf(prices, df_1d, senkou_span_a.values)
    senkou_span_b_aligned = align_htf_to_ltf(prices, df_1d, senkou_span_b.values)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 52  # Wait for Ichimoku calculations
    
    for i in range(start_idx, n):
        # Skip if any required data is not available
        if (np.isnan(vwap_1w_aligned[i]) or np.isnan(tenkan_sen_aligned[i]) or 
            np.isnan(kijun_sen_aligned[i]) or np.isnan(senkou_span_a_aligned[i]) or 
            np.isnan(senkou_span_b_aligned[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        vwap = vwap_1w_aligned[i]
        tenkan = tenkan_sen_aligned[i]
        kijun = kijun_sen_aligned[i]
        span_a = senkou_span_a_aligned[i]
        span_b = senkou_span_b_aligned[i]
        
        # Ichimoku cloud boundaries (Senkou Span A and B form the cloud)
        upper_cloud = max(span_a, span_b)
        lower_cloud = min(span_a, span_b)
        
        if position == 0:
            # Long entry: Price above weekly VWAP AND above Ichimoku cloud (bullish alignment)
            if price > vwap and price > upper_cloud:
                signals[i] = 0.25
                position = 1
            # Short entry: Price below weekly VWAP AND below Ichimoku cloud (bearish alignment)
            elif price < vwap and price < lower_cloud:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Long exit: Price falls below weekly VWAP OR below Ichimoku cloud
            if price < vwap or price < lower_cloud:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: Price rises above weekly VWAP OR above Ichimoku cloud
            if price > vwap or price > upper_cloud:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals
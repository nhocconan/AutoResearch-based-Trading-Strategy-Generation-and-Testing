#!/usr/bin/env python3
"""
6h_bollinger_bandwidth_regime_v1
Hypothesis: Use 1d Bollinger Bandwidth percentile to detect market regime (trending vs ranging). In trending regime (BW > 60th percentile), trade 6h Donchian(20) breakouts in direction of 1d EMA50/EMA200 trend. In ranging regime (BW <= 60th), fade 6h Donchian(20) breaks with reversion to 6h VWAP. Volume confirmation required for all entries. This adapts to market conditions: trend following in strong moves, mean reversion in chop. Targets 15-25 trades/year via strict regime filter and volume confirmation.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "6h_bollinger_bandwidth_regime_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 200:
        return np.zeros(n)
    
    # Price data
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # 6h VWAP for mean reversion target
    typical_price = (high + low + close) / 3.0
    vwap_num = np.cumsum(typical_price * volume)
    vwap_den = np.cumsum(volume)
    vwap = vwap_num / vwap_den
    
    # 6h Donchian channels (20-period)
    high_roll = pd.Series(high).rolling(window=20, min_periods=20).max().values
    low_roll = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # 6h EMA20 for momentum filter in trending regime
    ema20 = pd.Series(close).ewm(span=20, min_periods=20, adjust=False).mean().values
    
    # 1d Bollinger Bandwidth for regime detection (20, 2)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # Calculate Bollinger Bands on 1d close
    close_1d = df_1d['close'].values
    sma_20 = pd.Series(close_1d).rolling(window=20, min_periods=20).mean().values
    std_20 = pd.Series(close_1d).rolling(window=20, min_periods=20).std().values
    upper_bb = sma_20 + 2 * std_20
    lower_bb = sma_20 - 2 * std_20
    bandwidth = (upper_bb - lower_bb) / sma_20  # Normalized bandwidth
    
    # 50-period percentile rank of bandwidth (regime threshold)
    bw_series = pd.Series(bandwidth)
    bw_percentile = bw_series.rolling(window=50, min_periods=50).apply(
        lambda x: pd.Series(x).rank(pct=True).iloc[-1] * 100, raw=False
    ).values
    
    # Align 1d indicators to 6h
    sma_20_6h = align_htf_to_ltf(prices, df_1d, sma_20)
    std_20_6h = align_htf_to_ltf(prices, df_1d, std_20)
    upper_bb_6h = align_htf_to_ltf(prices, df_1d, upper_bb)
    lower_bb_6h = align_htf_to_ltf(prices, df_1d, lower_bb)
    bw_percentile_6h = align_htf_to_ltf(prices, df_1d, bw_percentile)
    
    # 1d EMA50/EMA200 for trend direction
    ema50_1d = pd.Series(close_1d).ewm(span=50, min_periods=50, adjust=False).mean().values
    ema200_1d = pd.Series(close_1d).ewm(span=200, min_periods=200, adjust=False).mean().values
    ema50_1d_6h = align_htf_to_ltf(prices, df_1d, ema50_1d)
    ema200_1d_6h = align_htf_to_ltf(prices, df_1d, ema200_1d)
    
    # Volume confirmation (24-period average on 6h = 4 days)
    vol_ma = pd.Series(volume).rolling(window=24, min_periods=24).mean().values
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(200, n):
        # Skip if required data not available
        if (np.isnan(high_roll[i]) or np.isnan(low_roll[i]) or
            np.isnan(ema20[i]) or np.isnan(vwap[i]) or
            np.isnan(bw_percentile_6h[i]) or np.isnan(ema50_1d_6h[i]) or
            np.isnan(ema200_1d_6h[i]) or np.isnan(vol_ma[i]) or vol_ma[i] <= 0):
            signals[i] = 0.0
            continue
        
        # Volume confirmation: current volume > 1.5x 24-period average
        vol_confirm = volume[i] > 1.5 * vol_ma[i]
        
        # Regime: trending if bandwidth > 60th percentile
        is_trending = bw_percentile_6h[i] > 60
        
        if position == 1:  # Long position
            # Exit conditions
            exit_long = False
            if is_trending:
                # In trend: exit on Donchian lower band break or EMA20 cross down
                if close[i] < low_roll[i]:
                    exit_long = True
                elif ema20[i] < close[i] and ema20[i-1] >= close[i-1]:
                    exit_long = True
            else:
                # In range: exit at VWAP (mean reversion target)
                if close[i] >= vwap[i]:
                    exit_long = True
            
            if exit_long:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit conditions
            exit_short = False
            if is_trending:
                # In trend: exit on Donchian upper band break or EMA20 cross up
                if close[i] > high_roll[i]:
                    exit_short = True
                elif ema20[i] > close[i] and ema20[i-1] <= close[i-1]:
                    exit_short = True
            else:
                # In range: exit at VWAP (mean reversion target)
                if close[i] <= vwap[i]:
                    exit_short = True
            
            if exit_short:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat, look for entry
            if is_trending:
                # TRENDING REGIME: Donchian breakout with EMA20 filter
                long_entry = (close[i] > high_roll[i] and 
                             ema20[i] > close[i] and  # Price above EMA20 = uptrend
                             ema50_1d_6h[i] > ema200_1d_6h[i] and  # 1d uptrend
                             vol_confirm)
                short_entry = (close[i] < low_roll[i] and 
                              ema20[i] < close[i] and  # Price below EMA20 = downtrend
                              ema50_1d_6h[i] < ema200_1d_6h[i] and  # 1d downtrend
                              vol_confirm)
            else:
                # RANGING REGIME: Fade Donchian breaks with VWAP target
                long_entry = (close[i] < low_roll[i] and  # Break below lower band
                             close[i] < vwap[i] and      # Below VWAP = oversold
                             vol_confirm)
                short_entry = (close[i] > high_roll[i] and  # Break above upper band
                              close[i] > vwap[i] and      # Above VWAP = overbought
                              vol_confirm)
            
            if long_entry:
                position = 1
                signals[i] = 0.25
            elif short_entry:
                position = -1
                signals[i] = -0.25
    
    return signals
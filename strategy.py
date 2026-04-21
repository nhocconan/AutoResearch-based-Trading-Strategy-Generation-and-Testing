#!/usr/bin/env python3
"""
6h_1d_MultiTimeframe_Candlestick_Pattern_Strategy
Hypothesis: Use daily candle structure to identify institutional supply/demand zones, combined with 6h price action and volume confirmation. In bull markets: buy pullbacks to demand zones with bullish engulfing patterns. In bear markets: sell rallies to supply zones with bearish engulfing patterns. Low frequency (target: 12-37/year) to minimize fee drag. Works in both regimes by adapting to trend direction via daily close vs open.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Load daily data once for supply/demand zones and trend
    df_daily = get_htf_data(prices, '1d')
    if len(df_daily) < 2:
        return np.zeros(n)
    
    high_daily = df_daily['high'].values
    low_daily = df_daily['low'].values
    open_daily = df_daily['open'].values
    close_daily = df_daily['close'].values
    
    # Identify daily supply and demand zones
    # Demand zone: bullish candle (close > open) - area where buying occurred
    # Supply zone: bearish candle (close < open) - area where selling occurred
    bullish_daily = close_daily > open_daily
    bearish_daily = close_daily < open_daily
    
    # Demand zone: low of bullish candles
    demand_zone = np.where(bullish_daily, low_daily, np.nan)
    # Supply zone: high of bearish candles
    supply_zone = np.where(bearish_daily, high_daily, np.nan)
    
    # Forward fill zones until next opposite zone appears
    demand_zone_series = pd.Series(demand_zone)
    demand_zone_filled = demand_zone_series.ffill().bfill().values
    supply_zone_series = pd.Series(supply_zone)
    supply_zone_filled = supply_zone_series.ffill().bfill().values
    
    # Align zones to 6h timeframe
    demand_zone_aligned = align_htf_to_ltf(prices, df_daily, demand_zone_filled)
    supply_zone_aligned = align_htf_to_ltf(prices, df_daily, supply_zone_filled)
    
    # Daily trend: close vs open (bullish/bearish bias)
    daily_bias = np.where(bullish_daily, 1, np.where(bearish_daily, -1, 0))
    daily_bias_aligned = align_htf_to_ltf(prices, df_daily, daily_bias)
    
    # Main timeframe data (6h)
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    open_price = prices['open'].values
    volume = prices['volume'].values
    
    # Volume filter: current volume > 1.3x 20-period average (less strict for fewer trades)
    volume_avg = np.zeros_like(volume)
    for i in range(len(volume)):
        if i >= 20:
            volume_avg[i] = np.mean(volume[i-20:i])
        else:
            volume_avg[i] = np.mean(volume[:i+1]) if i > 0 else volume[i]
    volume_filter = volume > (1.3 * volume_avg)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(20, n):
        # Skip if NaN in critical values
        if (np.isnan(demand_zone_aligned[i]) or np.isnan(supply_zone_aligned[i]) or 
            np.isnan(daily_bias_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        price = close[i]
        demand = demand_zone_aligned[i]
        supply = supply_zone_aligned[i]
        bias = daily_bias_aligned[i]
        vol_ok = volume_filter[i]
        
        # 6h bullish engulfing: current bullish candle engulfs previous bearish candle
        bullish_engulfing = (close[i] > open_price[i]) and (open_price[i-1] > close[i-1]) and \
                           (close[i] > open_price[i-1]) and (open_price[i] < close[i-1])
        # 6h bearish engulfing: current bearish candle engulfs previous bullish candle
        bearish_engulfing = (close[i] < open_price[i]) and (open_price[i-1] < close[i-1]) and \
                           (close[i] < open_price[i-1]) and (open_price[i] > close[i-1])
        
        if position == 0:
            # Long: price at demand zone with bullish engulfing in bullish daily bias
            if bias > 0 and abs(price - demand) < (price * 0.005) and bullish_engulfing and vol_ok:
                signals[i] = 0.25
                position = 1
            # Short: price at supply zone with bearish engulfing in bearish daily bias
            elif bias < 0 and abs(price - supply) < (price * 0.005) and bearish_engulfing and vol_ok:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Long exit: price reaches supply zone or bearish engulfing forms
            if abs(price - supply) < (price * 0.005) or bearish_engulfing:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: price reaches demand zone or bullish engulfing forms
            if abs(price - demand) < (price * 0.005) or bullish_engulfing:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "6h_1d_MultiTimeframe_Candlestick_Pattern_Strategy"
timeframe = "6h"
leverage = 1.0
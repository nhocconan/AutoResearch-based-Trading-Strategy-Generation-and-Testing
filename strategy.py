#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h strategy using daily VWAP as dynamic support/resistance with volume confirmation and RSI filter.
# Long when price crosses above VWAP AND volume > 1.5x 20-period average AND RSI(14) > 50.
# Short when price crosses below VWAP AND volume > 1.5x 20-period average AND RSI(14) < 50.
# Exit when price crosses back below/above VWAP OR RSI crosses back to neutral zone.
# VWAP provides intraday mean reversion level, volume confirms institutional participation, RSI filters momentum.
# Target: 20-40 trades/year per symbol (80-160 total over 4 years) to balance opportunity and cost.

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Load daily data ONCE for VWAP and volume average
    df_daily = get_htf_data(prices, '1d')
    if len(df_daily) < 30:
        return np.zeros(n)
    
    high_daily = df_daily['high'].values
    low_daily = df_daily['low'].values
    close_daily = df_daily['close'].values
    volume_daily = df_daily['volume'].values
    
    # Calculate typical price and cumulative VWAP components
    typical_price_daily = (high_daily + low_daily + close_daily) / 3.0
    pv_daily = typical_price_daily * volume_daily
    
    # Cumulative sums for VWAP (reset daily)
    cum_pv = np.cumsum(pv_daily)
    cum_volume = np.cumsum(volume_daily)
    vwap_daily = cum_pv / cum_volume
    
    # Calculate 20-period average volume
    vol_ma_20 = np.full_like(volume_daily, np.nan)
    for i in range(19, len(volume_daily)):
        vol_ma_20[i] = np.mean(volume_daily[i-19:i+1])
    
    # Calculate RSI(14) on daily close
    delta = np.diff(close_daily, prepend=close_daily[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    
    # Wilder's smoothing for RSI
    def wilders_smoothing(data, period):
        result = np.full_like(data, np.nan)
        if len(data) < period:
            return result
        # First average
        result[period-1] = np.mean(data[:period])
        for i in range(period, len(data)):
            result[i] = (result[i-1] * (period-1) + data[i]) / period
        return result
    
    avg_gain = wilders_smoothing(gain, 14)
    avg_loss = wilders_smoothing(loss, 14)
    rs = np.where(avg_loss != 0, avg_gain / avg_loss, 0)
    rsi_daily = 100 - (100 / (1 + rs))
    
    # Align indicators to 4h timeframe
    vwap_aligned = align_htf_to_ltf(prices, df_daily, vwap_daily)
    vol_ma_20_aligned = align_htf_to_ltf(prices, df_daily, vol_ma_20)
    rsi_aligned = align_htf_to_ltf(prices, df_daily, rsi_daily)
    
    signals = np.zeros(n)
    position = 0
    position_size = 0.25  # 25% position size
    
    # Start after enough data for calculations
    start = max(30, 20)
    
    for i in range(start, n):
        # Skip if any critical data is NaN
        if (np.isnan(vwap_aligned[i]) or 
            np.isnan(vol_ma_20_aligned[i]) or
            np.isnan(rsi_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Volume ratio: current 4h volume vs daily 20-period average volume (scaled)
        # Approximate daily volume from 4h by scaling (6 periods per day)
        daily_volume_approx = volume[i] * 6  # Rough estimate of daily volume from 4h bar
        vol_ma_20_scaled = vol_ma_20_aligned[i] * 6  # Scale to comparable units
        volume_ratio = daily_volume_approx / vol_ma_20_scaled if vol_ma_20_scaled > 0 else 0
        
        if position == 0:
            # Look for VWAP cross entries with volume confirmation and RSI filter
            # Long: price crosses above VWAP AND volume > 1.5x average AND RSI > 50
            if (close[i] > vwap_aligned[i] and close[i-1] <= vwap_aligned[i-1] and 
                volume_ratio > 1.5 and 
                rsi_aligned[i] > 50):
                position = 1
                signals[i] = position_size
            # Short: price crosses below VWAP AND volume > 1.5x average AND RSI < 50
            elif (close[i] < vwap_aligned[i] and close[i-1] >= vwap_aligned[i-1] and 
                  volume_ratio > 1.5 and 
                  rsi_aligned[i] < 50):
                position = -1
                signals[i] = -position_size
            else:
                signals[i] = 0.0
        elif position == 1:
            # Exit long: price crosses below VWAP OR RSI drops below 40
            if (close[i] < vwap_aligned[i] and close[i-1] >= vwap_aligned[i-1]) or \
               (rsi_aligned[i] < 40):
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = position_size
        elif position == -1:
            # Exit short: price crosses above VWAP OR RSI rises above 60
            if (close[i] > vwap_aligned[i] and close[i-1] <= vwap_aligned[i-1]) or \
               (rsi_aligned[i] > 60):
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -position_size
    
    return signals

name = "4h_VWAP_Volume_RSI_v1"
timeframe = "4h"
leverage = 1.0
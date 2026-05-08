#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Volume-Weighted Price Action with Daily Trend Filter
# Uses volume-weighted average price (VWAP) deviation from daily VWAP as mean reversion signal.
# Long when price deviates significantly below daily VWAP with volume confirmation.
# Short when price deviates significantly above daily VWAP with volume confirmation.
# Daily trend filter (price above/below daily EMA20) prevents counter-trend trades.
# Volume spike (>2x 20-period average) confirms institutional participation.
# Designed to work in both bull and bear markets by fading extreme deviations from fair value.
# Target: 60-120 total trades over 4 years (15-30/year) to stay within optimal range.

name = "6h_VWAP_MeanReversion_TrendFilter"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get daily data for VWAP and trend filter
    df_daily = get_htf_data(prices, '1d')
    if len(df_daily) < 20:
        return np.zeros(n)
    
    # Calculate daily VWAP
    typical_price_daily = (df_daily['high'] + df_daily['low'] + df_daily['close']) / 3.0
    vwap_daily = (typical_price_daily * df_daily['volume']).cumsum() / df_daily['volume'].cumsum()
    vwap_daily = vwap_daily.values
    
    # Calculate daily EMA20 for trend filter
    close_daily = df_daily['close'].values
    ema20_daily = np.full(len(close_daily), np.nan)
    if len(close_daily) >= 20:
        ema20_daily[19] = np.mean(close_daily[:20])
        for i in range(20, len(close_daily)):
            ema20_daily[i] = (close_daily[i] * 2 + ema20_daily[i-1] * 18) / 20
    
    # Calculate daily volume average for volume filter
    vol_daily = df_daily['volume'].values
    vol_avg_20_daily = np.full(len(vol_daily), np.nan)
    if len(vol_daily) >= 20:
        for i in range(20, len(vol_daily)):
            vol_avg_20_daily[i] = np.mean(vol_daily[i-20:i])
    
    # Align daily indicators to 6h timeframe
    vwap_daily_aligned = align_htf_to_ltf(prices, df_daily, vwap_daily)
    ema20_daily_aligned = align_htf_to_ltf(prices, df_daily, ema20_daily)
    vol_avg_20_daily_aligned = align_htf_to_ltf(prices, df_daily, vol_avg_20_daily)
    
    # Calculate 6-period standard deviation of price deviation from VWAP (for z-score)
    price_dev = close - vwap_daily_aligned
    price_dev_ma = np.full(n, np.nan)
    price_dev_std = np.full(n, np.nan)
    
    if n >= 6:
        for i in range(5, n):
            window = price_dev[i-5:i+1]
            if not np.any(np.isnan(window)):
                price_dev_ma[i] = np.mean(window)
                price_dev_std[i] = np.std(window)
    
    # Pre-compute session filter (08-20 UTC)
    hours = pd.DatetimeIndex(prices['open_time']).hour
    in_session = (hours >= 8) & (hours <= 20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(20, 6)  # warmup for indicators
    
    for i in range(start_idx, n):
        # Skip if outside trading session
        if not in_session[i]:
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Skip if any required data is NaN
        if (np.isnan(vwap_daily_aligned[i]) or np.isnan(ema20_daily_aligned[i]) or
            np.isnan(vol_avg_20_daily_aligned[i]) or np.isnan(price_dev[i]) or
            np.isnan(price_dev_std[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Volume filter: current daily volume > 2x 20-period average
        vol_filter = False
        if not np.isnan(vol_avg_20_daily_aligned[i]):
            # Find current daily bar's volume
            idx_daily = 0
            while idx_daily < len(df_daily) and df_daily.iloc[idx_daily]['open_time'] <= prices.iloc[i]['open_time']:
                idx_daily += 1
            idx_daily -= 1  # last completed daily bar
            
            if idx_daily >= 0:
                vol_daily_current = df_daily.iloc[idx_daily]['volume']
                vol_filter = vol_daily_current > 2.0 * vol_avg_20_daily_aligned[i]
        
        # Calculate z-score of price deviation from VWAP
        if price_dev_std[i] > 0:
            z_score = (price_dev[i] - price_dev_ma[i]) / price_dev_std[i]
        else:
            z_score = 0
        
        # Determine trend direction
        bullish_trend = close[i] > ema20_daily_aligned[i]
        bearish_trend = close[i] < ema20_daily_aligned[i]
        
        if position == 0:
            # Look for entry: mean reversion with volume and trend filter
            # Long when price is significantly below VWAP in bullish trend
            long_condition = (
                z_score < -2.0 and  # price dev more than 2 std below mean
                bullish_trend and   # only long in uptrend
                vol_filter
            )
            
            # Short when price is significantly above VWAP in bearish trend
            short_condition = (
                z_score > 2.0 and   # price dev more than 2 std above mean
                bearish_trend and   # only short in downtrend
                vol_filter
            )
            
            if long_condition:
                signals[i] = 0.25
                position = 1
            elif short_condition:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: price returns to VWAP or trend changes
            if z_score > -0.5 or not bullish_trend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: price returns to VWAP or trend changes
            if z_score < 0.5 or not bearish_trend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals
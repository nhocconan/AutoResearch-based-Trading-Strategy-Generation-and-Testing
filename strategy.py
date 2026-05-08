#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h ADX Trend Strength + 1d VWAP Reversion + Volume Spike
# Uses daily VWAP as mean reversion target with ADX(14) > 25 to filter strong trends.
# Enters when price deviates from daily VWAP by >1.5% in trending markets with volume >1.5x average.
# Works in bull/bear by following trend direction while avoiding choppy conditions. Target: 20-40 trades/year.

name = "4h_ADX_Trend_VWAPReversion_Volume"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get daily data for VWAP and trend context
    df_daily = get_htf_data(prices, '1d')
    if len(df_daily) < 30:
        return np.zeros(n)
    
    # Calculate daily VWAP (Volume Weighted Average Price)
    typical_price_daily = (df_daily['high'].values + df_daily['low'].values + df_daily['close'].values) / 3
    volume_daily = df_daily['volume'].values
    vwap_daily = np.full(len(typical_price_daily), np.nan)
    cum_vol_price = np.zeros(len(typical_price_daily))
    cum_vol = np.zeros(len(typical_price_daily))
    
    for i in range(len(typical_price_daily)):
        cum_vol_price[i] = (cum_vol_price[i-1] if i > 0 else 0) + typical_price_daily[i] * volume_daily[i]
        cum_vol[i] = (cum_vol[i-1] if i > 0 else 0) + volume_daily[i]
        if cum_vol[i] > 0:
            vwap_daily[i] = cum_vol_price[i] / cum_vol[i]
    
    # Calculate daily ADX (14-period) for trend strength
    high_daily = df_daily['high'].values
    low_daily = df_daily['low'].values
    close_daily = df_daily['close'].values
    
    # True Range
    tr = np.maximum(high_daily[1:] - low_daily[1:], 
                    np.maximum(np.abs(high_daily[1:] - close_daily[:-1]),
                               np.abs(low_daily[1:] - close_daily[:-1])))
    tr = np.concatenate([[np.nan], tr])
    
    # Directional Movement
    up_move = high_daily[1:] - high_daily[:-1]
    down_move = low_daily[:-1] - low_daily[1:]
    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0)
    plus_dm = np.concatenate([[0], plus_dm])
    minus_dm = np.concatenate([[0], minus_dm])
    
    # Smoothed values
    def smooth_wilder(arr, period):
        smoothed = np.full_like(arr, np.nan)
        if len(arr) < period:
            return smoothed
        smoothed[period-1] = np.nansum(arr[:period])
        for i in range(period, len(arr)):
            smoothed[i] = smoothed[i-1] - (smoothed[i-1] / period) + arr[i]
        return smoothed
    
    atr = smooth_wilder(tr, 14)
    plus_di = 100 * smooth_wilder(plus_dm, 14) / atr
    minus_di = 100 * smooth_wilder(minus_dm, 14) / atr
    dx = 100 * np.abs(plus_di - minus_di) / (plus_di + minus_di)
    adx = smooth_wilder(dx, 14)
    
    # Calculate daily volume average for volume spike filter
    vol_avg_20_daily = np.full(len(volume_daily), np.nan)
    if len(volume_daily) >= 20:
        for i in range(20, len(volume_daily)):
            vol_avg_20_daily[i] = np.mean(volume_daily[i-20:i])
    
    # Align daily indicators to 4h timeframe
    vwap_daily_aligned = align_htf_to_ltf(prices, df_daily, vwap_daily)
    adx_daily_aligned = align_htf_to_ltf(prices, df_daily, adx)
    vol_avg_20_daily_aligned = align_htf_to_ltf(prices, df_daily, vol_avg_20_daily)
    
    # Pre-compute session filter (08-20 UTC)
    hours = pd.DatetimeIndex(prices['open_time']).hour
    in_session = (hours >= 8) & (hours <= 20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(30, 20)
    
    for i in range(start_idx, n):
        # Skip if outside trading session
        if not in_session[i]:
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Skip if any required data is NaN
        if (np.isnan(vwap_daily_aligned[i]) or np.isnan(adx_daily_aligned[i]) or
            np.isnan(vol_avg_20_daily_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Volume spike: current 4h volume > 1.5x 20-period average of daily volume
        vol_spike = volume[i] > 1.5 * vol_avg_20_daily_aligned[i]
        
        # Price deviation from VWAP
        if vwap_daily_aligned[i] > 0:
            dev_pct = (close[i] - vwap_daily_aligned[i]) / vwap_daily_aligned[i] * 100
        else:
            dev_pct = 0
        
        # Strong trend filter: ADX > 25
        strong_trend = adx_daily_aligned[i] > 25
        
        if position == 0:
            # Look for entry: mean reversion to VWAP in strong trend
            # Long when price is significantly below VWAP in uptrend
            long_condition = (
                dev_pct < -1.5 and    # price below VWAP by >1.5%
                strong_trend and      # strong trend present
                vol_spike             # volume spike for confirmation
            )
            
            # Short when price is significantly above VWAP in downtrend
            short_condition = (
                dev_pct > 1.5 and     # price above VWAP by >1.5%
                strong_trend and      # strong trend present
                vol_spike             # volume spike for confirmation
            )
            
            if long_condition:
                signals[i] = 0.25
                position = 1
            elif short_condition:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: price returns to VWAP or trend weakens
            if dev_pct > -0.5 or adx_daily_aligned[i] < 20:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: price returns to VWAP or trend weakens
            if dev_pct < 0.5 or adx_daily_aligned[i] < 20:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals
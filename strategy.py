#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h price action near weekly VWAP with daily trend filter and volume confirmation
# Uses weekly VWAP as dynamic support/resistance, filtered by daily EMA50 trend and volume > 1.5x 20-period average.
# Designed to capture mean-reversion bounces off weekly VWAP in ranging markets and trend continuations in trending markets.
# Target: 15-30 trades/year (60-120 total over 4 years). Works in bull/bear via trend filter and VWAP reversion logic.

name = "12h_WeeklyVWAP_DailyTrend_VolumeFilter"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get weekly data for VWAP calculation
    df_weekly = get_htf_data(prices, '1w')
    if len(df_weekly) < 10:
        return np.zeros(n)
    
    # Calculate weekly VWAP (typical price * volume cumulative / volume cumulative)
    typical_price_weekly = (df_weekly['high'].values + df_weekly['low'].values + df_weekly['close'].values) / 3
    vp_weekly = typical_price_weekly * df_weekly['volume'].values
    cum_vp_weekly = np.cumsum(vp_weekly)
    cum_volume_weekly = np.cumsum(df_weekly['volume'].values)
    vwap_weekly = np.divide(cum_vp_weekly, cum_volume_weekly, out=np.full_like(cum_vp_weekly, np.nan), where=cum_volume_weekly!=0)
    
    # Get daily data for EMA50 trend
    df_daily = get_htf_data(prices, '1d')
    if len(df_daily) < 50:
        return np.zeros(n)
    
    # Calculate daily EMA50 for trend
    close_daily = df_daily['close'].values
    ema50_daily = np.full(len(close_daily), np.nan)
    if len(close_daily) >= 50:
        ema50_daily[49] = np.mean(close_daily[:50])
        for i in range(50, len(close_daily)):
            ema50_daily[i] = (close_daily[i] * 2 + ema50_daily[i-1] * 48) / 50
    
    # Calculate 12h volume average for volume filter
    vol_avg_20 = np.full(n, np.nan)
    if n >= 20:
        for i in range(20, n):
            vol_avg_20[i] = np.mean(volume[i-20:i])
    
    # Align weekly VWAP to 12h timeframe
    vwap_weekly_aligned = align_htf_to_ltf(prices, df_weekly, vwap_weekly)
    
    # Align daily EMA50 to 12h timeframe
    ema50_daily_aligned = align_htf_to_ltf(prices, df_daily, ema50_daily)
    
    # Pre-compute session filter (08-20 UTC)
    hours = pd.DatetimeIndex(prices['open_time']).hour
    in_session = (hours >= 8) & (hours <= 20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(50, 20)  # warmup for indicators
    
    for i in range(start_idx, n):
        # Skip if outside trading session
        if not in_session[i]:
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Skip if any required data is NaN
        if np.isnan(vwap_weekly_aligned[i]) or np.isnan(ema50_daily_aligned[i]) or np.isnan(vol_avg_20[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Find current weekly bar's VWAP (last completed weekly bar)
        vwap_weekly_current = np.nan
        if not np.isnan(vwap_weekly_aligned[i]):
            idx_weekly = 0
            while idx_weekly < len(df_weekly) and df_weekly.iloc[idx_weekly]['open_time'] <= prices.iloc[i]['open_time']:
                idx_weekly += 1
            idx_weekly -= 1  # last completed weekly bar
            
            if idx_weekly >= 0:
                vwap_weekly_current = df_weekly.iloc[idx_weekly]['vwap'] if 'vwap' in df_weekly.columns else vwap_weekly_aligned[i]
        
        # Find current daily bar's EMA50 (last completed daily bar)
        ema50_daily_current = np.nan
        if not np.isnan(ema50_daily_aligned[i]):
            idx_daily = 0
            while idx_daily < len(df_daily) and df_daily.iloc[idx_daily]['open_time'] <= prices.iloc[i]['open_time']:
                idx_daily += 1
            idx_daily -= 1  # last completed daily bar
            
            if idx_daily >= 0:
                ema50_daily_current = ema50_daily_aligned[i]
        
        if np.isnan(vwap_weekly_current) or np.isnan(ema50_daily_current):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Calculate deviation from weekly VWAP as percentage
        vwap_dev_pct = (close[i] - vwap_weekly_current) / vwap_weekly_current * 100
        
        # Check conditions
        price_above_ema = close[i] > ema50_daily_current
        price_below_ema = close[i] < ema50_daily_current
        vol_filter = volume[i] > 1.5 * vol_avg_20[i]
        
        if position == 0:
            # Look for entry: mean reversion off weekly VWAP with trend and volume filter
            # Long when price is below VWAP in uptrend, short when above VWAP in downtrend
            if vwap_dev_pct < -1.0 and price_above_ema and vol_filter:  # 1% below VWAP in uptrend
                signals[i] = 0.25
                position = 1
            elif vwap_dev_pct > 1.0 and price_below_ema and vol_filter:  # 1% above VWAP in downtrend
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: price returns to VWAP or trend fails or volume drops
            if abs(vwap_dev_pct) < 0.5 or not price_above_ema or not vol_filter:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: price returns to VWAP or trend fails or volume drops
            if abs(vwap_dev_pct) < 0.5 or not price_below_ema or not vol_filter:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals
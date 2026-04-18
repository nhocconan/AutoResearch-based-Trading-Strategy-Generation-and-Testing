#!/usr/bin/env python3
"""
1d_1w_WeeklyTrend_With_OpenInterest_Momentum_v1
Hypothesis: Use weekly price action above/below 20-week EMA for trend direction, combined with daily open interest momentum and price position within weekly range for entry timing. 
Go long when price is above weekly EMA20 AND daily OI increasing AND price in upper half of weekly range. 
Go short when price is below weekly EMA20 AND daily OI decreasing AND price in lower half of weekly range.
Uses volatility filter (weekly ATR < 50-day ATR median) to avoid choppy markets.
Target: 10-20 trades/year by requiring multiple confluence factors. Designed to work in bull markets via trend following and in bear via short signals.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    open_interest = prices.get('open_interest', np.zeros(n)).values  # Some symbols may not have OI
    
    # Get weekly data for trend and range
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 30:  # Need enough weeks for EMA20
        return np.zeros(n)
    
    close_1w = df_1w['close'].values
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    
    # Weekly EMA(20) for trend
    ema_period = 20
    ema_1w = np.full_like(close_1w, np.nan)
    if len(close_1w) >= ema_period:
        ema_1w[ema_period-1] = np.mean(close_1w[:ema_period])
        for i in range(ema_period, len(close_1w)):
            ema_1w[i] = (close_1w[i] * 2/(ema_period+1)) + (ema_1w[i-1] * (ema_period-1)/(ema_period+1))
    
    # Weekly ATR for volatility filter
    atr_period = 14
    tr_1w = np.maximum(
        high_1w[1:] - low_1w[1:],
        np.maximum(
            np.abs(high_1w[1:] - close_1w[:-1]),
            np.abs(low_1w[1:] - close_1w[:-1])
        )
    )
    tr_1w = np.concatenate([[np.nan], tr_1w])
    atr_1w = np.full_like(tr_1w, np.nan)
    if len(tr_1w) >= atr_period:
        atr_1w[atr_period-1] = np.nanmean(tr_1w[1:atr_period])
        for i in range(atr_period, len(tr_1w)):
            atr_1w[i] = (atr_1w[i-1] * (atr_period-1) + tr_1w[i]) / atr_period
    
    # Weekly range position (0 = low, 1 = high)
    range_1w = high_1w - low_1w
    range_1w = np.where(range_1w == 0, 1, range_1w)  # Avoid division by zero
    position_in_weekly_range = (close_1w - low_1w) / range_1w
    
    # Align weekly indicators to daily
    ema_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_1w)
    atr_1w_aligned = align_htf_to_ltf(prices, df_1w, atr_1w)
    position_in_weekly_range_aligned = align_htf_to_ltf(prices, df_1w, position_in_weekly_range)
    
    # Get 50-day ATR for volatility comparison (using daily data)
    tr_daily = np.maximum(
        high[1:] - low[1:],
        np.maximum(
            np.abs(high[1:] - close[:-1]),
            np.abs(low[1:] - close[:-1])
        )
    )
    tr_daily = np.concatenate([[np.nan], tr_daily])
    atr_50d = np.full_like(tr_daily, np.nan)
    atr_period_50 = 50
    if len(tr_daily) >= atr_period_50:
        atr_50d[atr_period_50-1] = np.nanmean(tr_daily[1:atr_period_50])
        for i in range(atr_period_50, len(tr_daily)):
            atr_50d[i] = (atr_50d[i-1] * (atr_period_50-1) + tr_daily[i]) / atr_period_50
    
    # Daily open interest momentum (change from previous day)
    oi_change = np.diff(open_interest, prepend=open_interest[0])
    oi_ma_period = 10
    oi_ma = np.full_like(oi_change, np.nan)
    if len(oi_change) >= oi_ma_period:
        for i in range(oi_ma_period, len(oi_change)):
            oi_ma[i] = np.mean(oi_change[i-oi_ma_period:i])
    oi_increasing = oi_change > oi_ma
    
    # Volatility filter: weekly ATR < 50-day ATR median (use 30-period median of ratio)
    vol_ratio = atr_1w_aligned / atr_50d
    vol_ratio_median_period = 30
    vol_ratio_median = np.full_like(vol_ratio, np.nan)
    if len(vol_ratio) >= vol_ratio_median_period:
        for i in range(vol_ratio_median_period, len(vol_ratio)):
            vol_ratio_median[i] = np.nanmedian(vol_ratio[i-vol_ratio_median_period:i])
    low_volatility = vol_ratio < vol_ratio_median
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(ema_period, atr_period, oi_ma_period, vol_ratio_median_period) + 1
    
    for i in range(start_idx, n):
        # Skip if any required data is not available
        if (np.isnan(ema_1w_aligned[i]) or np.isnan(position_in_weekly_range_aligned[i]) or
            np.isnan(oi_increasing[i]) if i < len(oi_increasing) else True or
            np.isnan(low_volatility[i])):
            signals[i] = 0.0
            continue
        
        # Get current values
        price = close[i]
        ema = ema_1w_aligned[i]
        pos_in_range = position_in_weekly_range_aligned[i]
        oi_momentum = oi_increasing[i] if i < len(oi_increasing) else False
        vol_filter = low_volatility[i]
        
        if position == 0 and vol_filter:
            # Long: above weekly EMA20, OI increasing, in upper half of weekly range
            if price > ema and oi_momentum and pos_in_range > 0.5:
                signals[i] = 0.25
                position = 1
            # Short: below weekly EMA20, OI decreasing, in lower half of weekly range
            elif price < ema and not oi_momentum and pos_in_range < 0.5:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Long exit: price crosses below weekly EMA20 OR OI turns negative
            if price < ema or (i < len(oi_increasing) and not oi_increasing[i]):
                signals[i] = -0.25  # reverse to short
                position = -1
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: price crosses above weekly EMA20 OR OI turns positive
            if price > ema or (i < len(oi_increasing) and oi_increasing[i]):
                signals[i] = 0.25  # reverse to long
                position = 1
            else:
                signals[i] = -0.25
    
    return signals

name = "1d_1w_WeeklyTrend_With_OpenInterest_Momentum_v1"
timeframe = "1d"
leverage = 1.0
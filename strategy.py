#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "12h_Choppiness_Regime_ADX_Trend_Filter"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get daily data for ADX and Choppiness Index
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # Get weekly data for trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 50:
        return np.zeros(n)
    
    # Daily indicators
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # ADX calculation (14-period)
    def calculate_adx(high, low, close, period=14):
        if len(high) < period + 1:
            return np.full(len(high), np.nan), np.full(len(high), np.nan), np.full(len(high), np.nan)
        
        # True Range
        tr1 = high[1:] - low[1:]
        tr2 = np.abs(high[1:] - close[:-1])
        tr3 = np.abs(low[1:] - close[:-1])
        tr = np.concatenate([[np.nan], np.maximum(tr1, np.maximum(tr2, tr3))])
        
        # Directional Movement
        dm_plus = np.where((high[1:] - high[:-1]) > (low[:-1] - low[1:]), np.maximum(high[1:] - high[:-1], 0), 0)
        dm_minus = np.where((low[:-1] - low[1:]) > (high[1:] - high[:-1]), np.maximum(low[:-1] - low[1:], 0), 0)
        dm_plus = np.concatenate([[np.nan], dm_plus])
        dm_minus = np.concatenate([[np.nan], dm_minus])
        
        # Smoothed values
        atr = np.zeros_like(tr)
        dm_plus_smooth = np.zeros_like(dm_plus)
        dm_minus_smooth = np.zeros_like(dm_minus)
        
        # Initial values
        atr[period] = np.nanmean(tr[1:period+1])
        dm_plus_smooth[period] = np.nanmean(dm_plus[1:period+1])
        dm_minus_smooth[period] = np.nanmean(dm_minus[1:period+1])
        
        # Wilder's smoothing
        for i in range(period + 1, len(tr)):
            atr[i] = (atr[i-1] * (period - 1) + tr[i]) / period
            dm_plus_smooth[i] = (dm_plus_smooth[i-1] * (period - 1) + dm_plus[i]) / period
            dm_minus_smooth[i] = (dm_minus_smooth[i-1] * (period - 1) + dm_minus[i]) / period
        
        # Directional Indicators
        di_plus = 100 * dm_plus_smooth / atr
        di_minus = 100 * dm_minus_smooth / atr
        
        # ADX
        dx = np.where((di_plus + di_minus) > 0, 100 * np.abs(di_plus - di_minus) / (di_plus + di_minus), 0)
        adx = np.zeros_like(dx)
        adx[2*period] = np.nanmean(dx[period:2*period+1])
        for i in range(2*period + 1, len(dx)):
            adx[i] = (adx[i-1] * (period - 1) + dx[i]) / period
        
        return adx, di_plus, di_minus
    
    adx, di_plus, di_minus = calculate_adx(high_1d, low_1d, close_1d, 14)
    
    # Choppiness Index (14-period)
    def calculate_choppiness(high, low, close, period=14):
        if len(high) < period + 1:
            return np.full(len(high), np.nan)
        
        # True Range
        tr1 = high[1:] - low[1:]
        tr2 = np.abs(high[1:] - close[:-1])
        tr3 = np.abs(low[1:] - close[:-1])
        tr = np.concatenate([[np.nan], np.maximum(tr1, np.maximum(tr2, tr3))])
        
        # Sum of ATR
        atr_sum = np.zeros_like(tr)
        for i in range(1, len(tr)):
            if i < period:
                atr_sum[i] = np.nan
            else:
                atr_sum[i] = np.nansum(tr[i-period+1:i+1])
        
        # Highest high and lowest low over period
        hh = np.zeros_like(high)
        ll = np.zeros_like(high)
        for i in range(len(high)):
            if i < period:
                hh[i] = np.nan
                ll[i] = np.nan
            else:
                hh[i] = np.nanmax(high[i-period+1:i+1])
                ll[i] = np.nanmin(low[i-period+1:i+1])
        
        # Choppiness
        chop = np.zeros_like(close)
        for i in range(len(close)):
            if i < period or np.isnan(atr_sum[i]) or np.isnan(hh[i]) or np.isnan(ll[i]) or hh[i] == ll[i]:
                chop[i] = np.nan
            else:
                chop[i] = 100 * np.log10(atr_sum[i] / (hh[i] - ll[i])) / np.log10(period)
        
        return chop
    
    chop = calculate_choppiness(high_1d, low_1d, close_1d, 14)
    
    # Weekly trend filter: EMA50
    ema50_1w = pd.Series(close_1w := df_1w['close'].values).ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Align indicators to 12h timeframe
    adx_aligned = align_htf_to_ltf(prices, df_1d, adx)
    chop_aligned = align_htf_to_ltf(prices, df_1d, chop)
    ema50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema50_1w)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 100  # warmup for indicators
    
    for i in range(start_idx, n):
        # Skip if any critical data is NaN
        if (np.isnan(adx_aligned[i]) or np.isnan(chop_aligned[i]) or 
            np.isnan(ema50_1w_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Regime detection
        is_trending = adx_aligned[i] > 25
        is_choppy = chop_aligned[i] > 61.8
        
        # Price relative to weekly EMA50
        price_above_weekly_ema = close[i] > ema50_1w_aligned[i]
        price_below_weekly_ema = close[i] < ema50_1w_aligned[i]
        
        if position == 0:
            # Long entry: trending market + price above weekly EMA50
            if is_trending and price_above_weekly_ema:
                signals[i] = 0.25
                position = 1
            # Short entry: trending market + price below weekly EMA50
            elif is_trending and price_below_weekly_ema:
                signals[i] = -0.25
                position = -1
            # In choppy market, mean revert at extremes
            elif is_choppy:
                # Simple mean reversion: buy when price is low, sell when high
                # Using daily close relative to its recent range
                if i >= 20:  # need some lookback for daily context
                    # Use daily close from aligned data (approximate)
                    daily_close_approx = close[i]  # simplification for 12h
                    # In practice would use aligned daily close, but for simplicity use price
                    if price_above_weekly_ema and chop_aligned[i] > 70:  # overbought in chop
                        signals[i] = -0.25
                        position = -1
                    elif price_below_weekly_ema and chop_aligned[i] > 70:  # oversold in chop
                        signals[i] = 0.25
                        position = 1
        elif position == 1:
            # Long exit: trend weakens or price crosses below weekly EMA50
            if not is_trending or price_below_weekly_ema:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: trend weakens or price crosses above weekly EMA50
            if not is_trending or price_above_weekly_ema:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

# Hypothesis: Combines ADX trend strength with Choppiness Index regime detection.
# In trending markets (ADX > 25), follow the weekly EMA50 trend.
# In choppy markets (Choppiness > 61.8), mean revert at extremes.
# Weekly EMA50 provides multi-timeframe trend filter to avoid counter-trend trades.
# Designed for 12h timeframe to target 12-37 trades per year (50-150 total over 4 years).
# Uses discrete position sizes (0.25) to minimize fee churn from signal changes.
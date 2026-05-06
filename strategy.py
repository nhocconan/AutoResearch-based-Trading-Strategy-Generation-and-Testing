#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h ADX(25) trend strength filter with 1d Bollinger Band mean reversion
# Uses 1d Bollinger Bands (20,2) for mean reversion signals, 4h ADX(25) for trend filter
# Long when price touches lower BB in weak trend (ADX<25), short when price touches upper BB in weak trend
# Volatility filter: only trade when BB width > 5th percentile (avoid squeeze)
# Strong trends (ADX>25) use inverse logic: buy upper BB breakout, sell lower BB breakdown
# Designed for low trade frequency (20-40/year) with clear regime-dependent logic
# Works in bull/bear: mean reversion in ranging markets, breakout in trending markets

name = "4h_ADX25_1dBBands_RegimeSwitch_v1"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Calculate HTF data ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    
    if len(df_1d) < 30:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate 1d Bollinger Bands (20,2)
    sma_20 = pd.Series(close_1d).rolling(window=20, min_periods=20).mean().values
    std_20 = pd.Series(close_1d).rolling(window=20, min_periods=20).std().values
    bb_upper = sma_20 + (2 * std_20)
    bb_lower = sma_20 - (2 * std_20)
    bb_width = bb_upper - bb_lower
    
    # Calculate BB width percentile for volatility filter (avoid low volatility)
    # Use 250-day lookback for percentile (approx 1 year)
    bb_width_percentile = pd.Series(bb_width).rolling(window=250, min_periods=50).apply(
        lambda x: pd.Series(x).rank(pct=True).iloc[-1] if len(x) > 0 else 0.5, raw=False
    ).values
    
    # Calculate 4h ADX(25) trend filter
    # TR = max(high-low, |high-prev_close|, |low-prev_close|)
    tr1 = np.abs(high[1:] - low[1:])
    tr2 = np.abs(high[1:] - close[:-1])
    tr3 = np.abs(low[1:] - close[:-1])
    tr = np.concatenate([[np.nan], np.maximum(tr1, np.maximum(tr2, tr3))])
    
    # +DM = max(high - prev_high, 0) if high - prev_high > prev_low - low else 0
    dm_plus = np.where((high[1:] - high[:-1]) > (low[:-1] - low[1:]), 
                       np.maximum(high[1:] - high[:-1], 0), 0)
    dm_plus = np.concatenate([[0], dm_plus])
    
    # -DM = max(prev_low - low, 0) if prev_low - low > high - prev_high else 0
    dm_minus = np.where((low[:-1] - low[1:]) > (high[1:] - high[:-1]), 
                        np.maximum(low[:-1] - low[1:], 0), 0)
    dm_minus = np.concatenate([[0], dm_minus])
    
    # Wilder's smoothing function
    def wilder_smooth(data, period):
        result = np.full_like(data, np.nan)
        alpha = 1.0 / period
        if len(data) >= period:
            result[period-1] = np.nanmean(data[:period])
            for i in range(period, len(data)):
                result[i] = result[i-1] + alpha * (data[i] - result[i-1])
        return result
    
    atr_4h = wilder_smooth(tr, 25)
    dm_plus_smooth = wilder_smooth(dm_plus, 25)
    dm_minus_smooth = wilder_smooth(dm_minus, 25)
    
    # DI+ = 100 * smoothed +DM / ATR, DI- = 100 * smoothed -DM / ATR
    di_plus = np.where(atr_4h != 0, 100 * dm_plus_smooth / atr_4h, 0)
    di_minus = np.where(atr_4h != 0, 100 * dm_minus_smooth / atr_4h, 0)
    
    # DX = 100 * |DI+ - DI-| / (DI+ + DI-)
    dx = np.where((di_plus + di_minus) != 0, 100 * np.abs(di_plus - di_minus) / (di_plus + di_minus), 0)
    
    # ADX = smoothed DX
    adx_4h = wilder_smooth(dx, 25)
    
    # Align HTF indicators to 4h timeframe (primary)
    bb_upper_aligned = align_htf_to_ltf(prices, df_1d, bb_upper)
    bb_lower_aligned = align_htf_to_ltf(prices, df_1d, bb_lower)
    bb_width_percentile_aligned = align_htf_to_ltf(prices, df_1d, bb_width_percentile)
    
    # Pre-compute session filter (08-20 UTC)
    hours = pd.DatetimeIndex(prices["open_time"]).hour
    session_filter = (hours >= 8) & (hours <= 20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(50, n):
        # Skip if any critical value is NaN or outside session
        if (np.isnan(adx_4h[i]) or np.isnan(bb_upper_aligned[i]) or 
            np.isnan(bb_lower_aligned[i]) or np.isnan(bb_width_percentile_aligned[i]) or
            not session_filter[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Volatility filter: only trade when BB width > 30th percentile (avoid low volatility squeeze)
        vol_filter = bb_width_percentile_aligned[i] > 0.30
        
        if position == 0:
            # Regime-based entry logic
            if adx_4h[i] < 25:  # Weak trend/ranging market -> mean reversion
                if vol_filter:
                    # Long at lower BB, short at upper BB
                    if close[i] <= bb_lower_aligned[i]:
                        signals[i] = 0.25
                        position = 1
                    elif close[i] >= bb_upper_aligned[i]:
                        signals[i] = -0.25
                        position = -1
            else:  # Strong trend -> breakout/continuation
                if vol_filter:
                    # Buy strength, sell weakness
                    if close[i] > bb_upper_aligned[i]:
                        signals[i] = 0.25
                        position = 1
                    elif close[i] < bb_lower_aligned[i]:
                        signals[i] = -0.25
                        position = -1
        elif position == 1:
            # Exit long: price crosses above SMA (mean reversion) or below BB (stop)
            if close[i] >= sma_20[-1] if len(sma_20) > 0 else bb_upper_aligned[i] or \
               close[i] <= bb_lower_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: price crosses below SMA (mean reversion) or above BB (stop)
            if close[i] <= sma_20[-1] if len(sma_20) > 0 else bb_lower_aligned[i] or \
               close[i] >= bb_upper_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals
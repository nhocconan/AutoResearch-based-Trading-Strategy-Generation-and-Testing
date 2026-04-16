#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h ADX trend strength + weekly Bollinger Band squeeze + price action
# Long when ADX(14) > 25 (trending) AND price > BB(20,2) upper band AND weekly close > weekly open
# Short when ADX(14) > 25 (trending) AND price < BB(20,2) lower band AND weekly close < weekly open
# Exit when ADX < 20 (range) OR opposite BB band touch
# Uses weekly trend filter to avoid counter-trend trades in ranging markets
# Target: 60-120 total trades over 4 years (15-30/year) to balance opportunity and fee drag

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    
    # === Weekly trend filter (close > open = bullish week) ===
    df_1w = get_htf_data(prices, '1w')
    weekly_open = df_1w['open'].values
    weekly_close = df_1w['close'].values
    weekly_bullish = weekly_close > weekly_open  # True for bullish weekly candle
    weekly_bullish_aligned = align_htf_to_ltf(prices, df_1w, weekly_bullish.astype(float))
    
    # === 6h ADX trend strength ===
    # Calculate True Range
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr2[0] = tr1[0]
    tr3[0] = tr1[0]
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    
    # Calculate Directional Movement
    dm_plus = np.where((high - np.roll(high, 1)) > (np.roll(low, 1) - low), 
                       np.maximum(high - np.roll(high, 1), 0), 0)
    dm_minus = np.where((np.roll(low, 1) - low) > (high - np.roll(high, 1)), 
                        np.maximum(np.roll(low, 1) - low, 0), 0)
    dm_plus[0] = 0
    dm_minus[0] = 0
    
    # Smooth with Wilder's smoothing (equivalent to EMA with alpha=1/period)
    def wilder_smooth(data, period):
        result = np.full_like(data, np.nan)
        if len(data) >= period:
            # First value is simple average
            result[period-1] = np.nanmean(data[:period])
            # Subsequent values: smoothed = prev * (1 - 1/period) + current * (1/period)
            alpha = 1.0 / period
            for i in range(period, len(data)):
                if not np.isnan(result[i-1]):
                    result[i] = result[i-1] * (1 - alpha) + data[i] * alpha
        return result
    
    atr = wilder_smooth(tr, 14)
    dm_plus_smooth = wilder_smooth(dm_plus, 14)
    dm_minus_smooth = wilder_smooth(dm_minus, 14)
    
    # Calculate DI+ and DI-
    di_plus = np.where(atr > 0, 100 * dm_plus_smooth / atr, 0)
    di_minus = np.where(atr > 0, 100 * dm_minus_smooth / atr, 0)
    
    # Calculate DX and ADX
    dx = np.where((di_plus + di_minus) > 0, 100 * np.abs(di_plus - di_minus) / (di_plus + di_minus), 0)
    adx = wilder_smooth(dx, 14)
    
    # === 6h Bollinger Bands (20, 2) ===
    bb_period = 20
    bb_std = 2
    sma = pd.Series(close).rolling(window=bb_period, min_periods=bb_period).mean().values
    bb_std_dev = pd.Series(close).rolling(window=bb_period, min_periods=bb_period).std().values
    upper_band = sma + (bb_std_dev * bb_std)
    lower_band = sma - (bb_std_dev * bb_std)
    
    signals = np.zeros(n)
    
    # Warmup: need enough data for all indicators
    warmup = 50
    
    for i in range(warmup, n):
        # Skip if any data is NaN
        if (np.isnan(adx[i]) or 
            np.isnan(weekly_bullish_aligned[i]) or
            np.isnan(upper_band[i]) or
            np.isnan(lower_band[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        adx_val = adx[i]
        weekly_bull = weekly_bullish_aligned[i] > 0.5  # Convert back to boolean
        upper = upper_band[i]
        lower = lower_band[i]
        
        # === ENTRY LOGIC ===
        # Strong trend (ADX > 25) + price at Bollinger Band + weekly trend alignment
        if adx_val > 25:
            # Long: price touches/breaks upper band AND weekly bullish
            if price >= upper and weekly_bull:
                signals[i] = 0.25
            # Short: price touches/breaks lower band AND weekly bearish
            elif price <= lower and not weekly_bull:
                signals[i] = -0.25
            else:
                # Hold flat or previous signal
                signals[i] = signals[i-1] if i > 0 else 0
        else:
            # Weak trend (ADX < 25) - exit or stay flat
            signals[i] = 0.0
    
    return signals

name = "6h_ADX25_WeeklyTrend_BB20_2_Breakout"
timeframe = "6h"
leverage = 1.0
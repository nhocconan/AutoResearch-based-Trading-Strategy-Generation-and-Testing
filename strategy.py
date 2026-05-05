#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Volume-Weighted MACD + 1d ADX Trend Filter + Bollinger Band Mean Reversion
# Long when: MACD histogram crosses above zero AND price < lower BB(20,2) AND 1d ADX > 25 (trending market)
# Short when: MACD histogram crosses below zero AND price > upper BB(20,2) AND 1d ADX > 25
# Exit when MACD histogram crosses zero in opposite direction OR price crosses middle BB(20)
# Uses volume-weighted MACD to filter weak moves, ADX to ensure trending conditions, BB for mean reversion entries
# Targets 60-120 trades over 4 years (15-30/year) to minimize fee drag while capturing sustained moves
# Works in both bull and bear markets by only trading in direction of 1d trend (ADX > 25)

name = "6h_VolWeighted_MACD_ADX25_BB_MeanRev"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Get 1d data ONCE before loop for ADX trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    # Calculate 1d ADX for trend filter (14-period)
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # True Range
    tr1 = np.abs(high_1d[1:] - low_1d[1:])
    tr2 = np.abs(high_1d[1:] - close_1d[:-1])
    tr3 = np.abs(low_1d[1:] - close_1d[:-1])
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr = np.concatenate([[np.nan], tr])  # align with index
    
    # Directional Movement
    up_move = high_1d[1:] - high_1d[:-1]
    down_move = low_1d[:-1] - low_1d[1:]
    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0.0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0.0)
    plus_dm = np.concatenate([[np.nan], plus_dm])
    minus_dm = np.concatenate([[np.nan], minus_dm])
    
    # Smoothed values (Wilder's smoothing = EMA with alpha=1/period)
    def wilders_smoothing(data, period):
        result = np.full_like(data, np.nan)
        if len(data) < period:
            return result
        # First value is simple average
        result[period-1] = np.nanmean(data[1:period])
        # Subsequent values: Wilder's smoothing
        alpha = 1.0 / period
        for i in range(period, len(data)):
            if np.isnan(result[i-1]):
                result[i] = np.nan
            else:
                result[i] = alpha * data[i] + (1 - alpha) * result[i-1]
        return result
    
    tr_14 = wilders_smoothing(tr, 14)
    plus_dm_14 = wilders_smoothing(plus_dm, 14)
    minus_dm_14 = wilders_smoothing(minus_dm, 14)
    
    # Avoid division by zero
    plus_di_14 = np.where(tr_14 != 0, (plus_dm_14 / tr_14) * 100, 0)
    minus_di_14 = np.where(tr_14 != 0, (minus_dm_14 / tr_14) * 100, 0)
    
    dx = np.where((plus_di_14 + minus_di_14) != 0, 
                  np.abs(plus_di_14 - minus_di_14) / (plus_di_14 + minus_di_14) * 100, 0)
    adx_14 = wilders_smoothing(dx, 14)
    adx_14_aligned = align_htf_to_ltf(prices, df_1d, adx_14)
    
    # 6h Bollinger Bands (20, 2)
    if len(close) >= 20:
        bb_ma = pd.Series(close).rolling(window=20, min_periods=20).mean().values
        bb_std = pd.Series(close).rolling(window=20, min_periods=20).std().values
        bb_upper = bb_ma + 2 * bb_std
        bb_lower = bb_ma - 2 * bb_std
        bb_middle = bb_ma
    else:
        bb_ma = bb_upper = bb_lower = bb_middle = np.full(n, np.nan)
    
    # 6h Volume-Weighted MACD
    if len(close) >= 34:
        # Volume-weighted prices for MACD calculation
        vw_close = close * volume
        vw_sum = pd.Series(volume).rolling(window=1, min_periods=1).sum().values
        vw_price = np.where(vw_sum != 0, vw_close / vw_sum, close)
        
        # Calculate EMAs on volume-weighted price
        ema_12 = pd.Series(vw_price).ewm(span=12, adjust=False, min_periods=12).mean().values
        ema_26 = pd.Series(vw_price).ewm(span=26, adjust=False, min_periods=26).mean().values
        macd_line = ema_12 - ema_26
        signal_line = pd.Series(macd_line).ewm(span=9, adjust=False, min_periods=9).mean().values
        macd_hist = macd_line - signal_line
    else:
        ema_12 = ema_26 = macd_line = signal_line = macd_hist = np.full(n, np.nan)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(50, n):
        # Skip if any value is NaN
        if (np.isnan(macd_hist[i]) or np.isnan(adx_14_aligned[i]) or 
            np.isnan(bb_upper[i]) or np.isnan(bb_lower[i]) or np.isnan(bb_middle[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Only trade when 1d ADX > 25 (trending market)
        if adx_14_aligned[i] > 25:
            if position == 0:
                # Long: MACD hist crosses above zero AND price < lower BB
                if i > 50 and macd_hist[i-1] <= 0 and macd_hist[i] > 0 and close[i] < bb_lower[i]:
                    signals[i] = 0.25
                    position = 1
                # Short: MACD hist crosses below zero AND price > upper BB
                elif i > 50 and macd_hist[i-1] >= 0 and macd_hist[i] < 0 and close[i] > bb_upper[i]:
                    signals[i] = -0.25
                    position = -1
            elif position == 1:
                # Exit long: MACD hist crosses below zero OR price > middle BB
                if macd_hist[i] < 0 or close[i] > bb_middle[i]:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = 0.25
            elif position == -1:
                # Exit short: MACD hist crosses above zero OR price < middle BB
                if macd_hist[i] > 0 or close[i] < bb_middle[i]:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = -0.25
        else:
            # In ranging market (ADX <= 25), stay flat
            if position != 0:
                signals[i] = 0.0
                position = 0
    
    return signals
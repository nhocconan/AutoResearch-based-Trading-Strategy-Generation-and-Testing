#!/usr/bin/env python3
# 4h_Camarilla_Pivot_Squeeze_Breakout_1dTrend_Volume
# Hypothesis: Camarilla pivot levels from daily timeframe identify key support/resistance.
# Breakouts above R3 or below S3 with volume expansion and daily trend alignment.
# Uses Bollinger Band squeeze (low volatility) as entry filter to avoid false breakouts.
# Works in bull markets (breakouts up) and bear markets (breakdowns down).
# Target: 20-50 total trades over 4 years (5-12/year).

name = "4h_Camarilla_Pivot_Squeeze_Breakout_1dTrend_Volume"
timeframe = "4h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def camarilla_pivot(high, low, close):
    """Calculate Camarilla pivot levels for given high, low, close"""
    range_val = high - low
    if range_val <= 0:
        return np.full_like(high, np.nan), np.full_like(high, np.nan), \
               np.full_like(high, np.nan), np.full_like(high, np.nan)
    R4 = close + range_val * 1.1 / 2
    R3 = close + range_val * 1.1 / 4
    R2 = close + range_val * 1.1 / 6
    R1 = close + range_val * 1.1 / 12
    S1 = close - range_val * 1.1 / 12
    S2 = close - range_val * 1.1 / 6
    S3 = close - range_val * 1.1 / 4
    S4 = close - range_val * 1.1 / 2
    return R1, R2, R3, R4, S1, S2, S3, S4

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)

    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values

    # Get daily data for Camarilla pivots and trend
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 10:
        return np.zeros(n)

    # Calculate Camarilla pivot levels on daily data
    R1, R2, R3, R4, S1, S2, S3, S4 = camarilla_pivot(
        df_1d['high'].values,
        df_1d['low'].values,
        df_1d['close'].values
    )
    
    # Use R3 and S3 as key resistance/support levels
    # Align to 4h timeframe with no extra delay (Camarilla levels are known at daily close)
    R3_aligned = align_htf_to_ltf(prices, df_1d, R3)
    S3_aligned = align_htf_to_ltf(prices, df_1d, S3)

    # Daily EMA34 trend filter (only needs completed daily candle)
    ema_34_1d = pd.Series(df_1d['close']).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)

    # Bollinger Band squeeze filter: low volatility environment
    # Calculate BB width on 4h data
    bb_ma = pd.Series(close).rolling(window=20, min_periods=20).mean()
    bb_std = pd.Series(close).rolling(window=20, min_periods=20).std()
    bb_upper = bb_ma + 2 * bb_std
    bb_lower = bb_ma - 2 * bb_std
    bb_width = (bb_upper - bb_lower) / bb_ma
    bb_width_values = bb_width.values
    
    # Squeeze condition: BB width below 20-period average of BB width
    bb_width_ma = pd.Series(bb_width_values).rolling(window=20, min_periods=20).mean()
    bb_width_ma_values = bb_width_ma.values
    squeeze_condition = bb_width_values < bb_width_ma_values

    # Volume confirmation: current volume > 1.3x average of last 20 periods
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_ok = volume > (1.3 * vol_ma)

    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short

    for i in range(50, n):
        # Skip if any required data is NaN
        if (np.isnan(R3_aligned[i]) or np.isnan(S3_aligned[i]) or
            np.isnan(ema_34_1d_aligned[i]) or np.isnan(squeeze_condition[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue

        # Trend filter from daily EMA34
        price_above_daily_ema = close[i] > ema_34_1d_aligned[i]
        price_below_daily_ema = close[i] < ema_34_1d_aligned[i]

        if position == 0:
            # LONG: Price breaks above R3 with volume expansion and volatility squeeze
            if (not np.isnan(R3_aligned[i]) and 
                close[i] > R3_aligned[i] and
                price_above_daily_ema and 
                volume_ok[i] and 
                squeeze_condition[i]):
                signals[i] = 0.25
                position = 1
            # SHORT: Price breaks below S3 with volume expansion and volatility squeeze
            elif (not np.isnan(S3_aligned[i]) and 
                  close[i] < S3_aligned[i] and
                  price_below_daily_ema and 
                  volume_ok[i] and 
                  squeeze_condition[i]):
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: Price breaks below S3 or trend turns down or volatility expands
            if ((not np.isnan(S3_aligned[i]) and close[i] < S3_aligned[i]) or 
                not price_above_daily_ema or 
                not squeeze_condition[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: Price breaks above R3 or trend turns up or volatility expands
            if ((not np.isnan(R3_aligned[i]) and close[i] > R3_aligned[i]) or 
                not price_below_daily_ema or 
                not squeeze_condition[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25

    return signals
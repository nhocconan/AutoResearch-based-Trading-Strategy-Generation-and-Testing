# 6h_Camarilla_R3_S3_Breakout_1dTrend_VolumeSpike
# Hypothesis: Fade at Camarilla R3/S3 levels (strong intraday support/resistance) with volume confirmation,
# but only in the direction of the 1d EMA34 trend to avoid counter-trend trades.
# In strong trends (ADX>25), breakout continuation at R4/S4 with volume spike.
# Works in bull (buy R3 bounce, sell R4 breakout) and bear (sell S3 bounce, buy S4 breakdown).
# Low frequency due to specific price levels and volume confirmation requirement.

name = "6h_Camarilla_R3_S3_Breakout_1dTrend_VolumeSpike"
timeframe = "6h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)

    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values

    # Get daily data for Camarilla levels and trend
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate daily Camarilla levels from previous day
    # Using prior day's OHLC to avoid look-ahead
    prev_high = df_1d['high'].shift(1).values
    prev_low = df_1d['low'].shift(1).values
    prev_close = df_1d['close'].shift(1).values
    
    # Camarilla levels
    range_ = prev_high - prev_low
    R3 = prev_close + range_ * 1.1 / 2
    S3 = prev_close - range_ * 1.1 / 2
    R4 = prev_close + range_ * 1.1
    S4 = prev_close - range_ * 1.1
    
    # Daily trend: EMA34
    ema34_1d = pd.Series(df_1d['close']).ewm(span=34, adjust=False, min_periods=34).mean().values
    
    # ADX for trend strength (using daily data)
    # Calculate +DM, -DM, TR
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    plus_dm = np.where((high_1d[1:] - high_1d[:-1]) > (low_1d[:-1] - low_1d[1:]), 
                       np.maximum(high_1d[1:] - high_1d[:-1], 0), 0)
    minus_dm = np.where((low_1d[:-1] - low_1d[1:]) > (high_1d[1:] - high_1d[:-1]), 
                        np.maximum(low_1d[:-1] - low_1d[1:], 0), 0)
    tr = np.maximum(
        high_1d[1:] - low_1d[1:],
        np.maximum(np.abs(high_1d[1:] - close_1d[:-1]), np.abs(low_1d[1:] - close_1d[:-1]))
    )
    
    # Smooth with Wilder's smoothing (alpha = 1/period)
    def wilders_smooth(data, period):
        result = np.zeros_like(data)
        result[period-1] = np.mean(data[:period])
        for i in range(period, len(data)):
            result[i] = (result[i-1] * (period-1) + data[i]) / period
        return result
    
    period = 14
    if len(plus_dm) >= period:
        plus_di = 100 * wilders_smooth(plus_dm, period) / wilders_smooth(tr[1:], period)
        minus_di = 100 * wilders_smooth(minus_dm, period) / wilders_smooth(tr[1:], period)
        dx = 100 * np.abs(plus_di - minus_di) / (plus_di + minus_di)
        adx = wilders_smooth(dx[~np.isnan(dx)], period) if len(dx[~np.isnan(dx)]) >= period else np.full_like(dx, np.nan)
        # Prepend NaN for the first element
        adx_full = np.full(len(high_1d), np.nan)
        adx_full[1:] = adx
        adx_1d = adx_full
    else:
        adx_1d = np.full_like(high_1d, np.nan)
    
    # Align daily indicators to 6h timeframe
    R3_aligned = align_htf_to_ltf(prices, df_1d, R3)
    S3_aligned = align_htf_to_ltf(prices, df_1d, S3)
    R4_aligned = align_htf_to_ltf(prices, df_1d, R4)
    S4_aligned = align_htf_to_ltf(prices, df_1d, S4)
    ema34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema34_1d)
    adx_1d_aligned = align_htf_to_ltf(prices, df_1d, adx_1d)
    
    # Volume spike: volume > 2.0 * 4-period average (1 day worth at 6h)
    vol_ma_4 = pd.Series(volume).rolling(window=4, min_periods=4).mean().values
    volume_spike = volume > 2.0 * vol_ma_4
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short

    for i in range(100, n):
        # Skip if any required value is NaN
        if (np.isnan(R3_aligned[i]) or 
            np.isnan(S3_aligned[i]) or 
            np.isnan(R4_aligned[i]) or 
            np.isnan(S4_aligned[i]) or 
            np.isnan(ema34_1d_aligned[i]) or 
            np.isnan(adx_1d_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue

        if position == 0:
            # Determine market regime
            is_trending = adx_1d_aligned[i] > 25
            
            if is_trending:
                # In trending market: breakout continuation at R4/S4
                # LONG: Close > R4 + uptrend + volume spike
                if close[i] > R4_aligned[i] and close[i] > ema34_1d_aligned[i] and volume_spike[i]:
                    signals[i] = 0.25
                    position = 1
                # SHORT: Close < S4 + downtrend + volume spike
                elif close[i] < S4_aligned[i] and close[i] < ema34_1d_aligned[i] and volume_spike[i]:
                    signals[i] = -0.25
                    position = -1
                else:
                    signals[i] = 0.0
            else:
                # In ranging market: fade at R3/S3
                # LONG: Close > R3 (bounce off support) + uptrend + volume spike
                if close[i] > R3_aligned[i] and close[i] > ema34_1d_aligned[i] and volume_spike[i]:
                    signals[i] = 0.25
                    position = 1
                # SHORT: Close < S3 (bounce off resistance) + downtrend + volume spike
                elif close[i] < S3_aligned[i] and close[i] < ema34_1d_aligned[i] and volume_spike[i]:
                    signals[i] = -0.25
                    position = -1
                else:
                    signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: Close below EMA34 OR reached opposite Camarilla level
            if is_trending := (adx_1d_aligned[i] > 25):
                # In trend: exit on trend reversal
                if close[i] < ema34_1d_aligned[i]:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = 0.25
            else:
                # In range: exit at opposite S3 level or EMA34
                if close[i] < S3_aligned[i] or close[i] < ema34_1d_aligned[i]:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: Close above EMA34 OR reached opposite Camarilla level
            if is_trending := (adx_1d_aligned[i] > 25):
                # In trend: exit on trend reversal
                if close[i] > ema34_1d_aligned[i]:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = -0.25
            else:
                # In range: exit at opposite R3 level or EMA34
                if close[i] > R3_aligned[i] or close[i] > ema34_1d_aligned[i]:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = -0.25

    return signals
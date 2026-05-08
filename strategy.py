#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1d Donchian breakout with weekly trend filter and volume confirmation.
# Trade breakouts of daily Donchian(20) when weekly trend aligns (trend follow)
# and volume confirms (>1.5x 20-period volume average).
# In ranging markets (weekly ADX < 25), fade at Donchian bands instead.
# Target: 20-50 total trades over 4 years (5-12/year) to minimize fee drag.

name = "1d_DonchianBreakout_1wTrend_Volume"
timeframe = "1d"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get weekly data for trend filter and ADX
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 30:
        return np.zeros(n)
    
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    volume_1w = df_1w['volume'].values
    
    # Weekly ADX(14) for trend strength
    def calculate_adx(high, low, close, period=14):
        plus_dm = np.zeros_like(high)
        minus_dm = np.zeros_like(high)
        tr = np.zeros_like(high)
        
        for i in range(1, len(high)):
            plus_dm[i] = max(0, high[i] - high[i-1])
            minus_dm[i] = max(0, low[i-1] - low[i])
            if plus_dm[i] > minus_dm[i]:
                minus_dm[i] = 0
            elif minus_dm[i] > plus_dm[i]:
                plus_dm[i] = 0
            else:
                plus_dm[i] = 0
                minus_dm[i] = 0
            
            tr[i] = max(
                high[i] - low[i],
                abs(high[i] - close[i-1]),
                abs(low[i] - close[i-1])
            )
        
        # Wilder's smoothing
        atr = np.zeros_like(high)
        atr[period] = np.mean(tr[1:period+1])
        for i in range(period+1, len(high)):
            atr[i] = (atr[i-1] * (period-1) + tr[i]) / period
        
        plus_di = 100 * (np.zeros_like(high))
        minus_di = 100 * (np.zeros_like(high))
        plus_dm_sm = np.zeros_like(high)
        minus_dm_sm = np.zeros_like(high)
        
        plus_dm_sm[period] = np.mean(plus_dm[1:period+1])
        minus_dm_sm[period] = np.mean(minus_dm[1:period+1])
        
        for i in range(period+1, len(high)):
            plus_dm_sm[i] = (plus_dm_sm[i-1] * (period-1) + plus_dm[i]) / period
            minus_dm_sm[i] = (minus_dm_sm[i-1] * (period-1) + minus_dm[i]) / period
        
        for i in range(period, len(high)):
            if atr[i] != 0:
                plus_di[i] = 100 * plus_dm_sm[i] / atr[i]
                minus_di[i] = 100 * minus_dm_sm[i] / atr[i]
        
        dx = np.zeros_like(high)
        for i in range(period, len(high)):
            if (plus_di[i] + minus_di[i]) != 0:
                dx[i] = 100 * abs(plus_di[i] - minus_di[i]) / (plus_di[i] + minus_di[i])
        
        adx = np.zeros_like(high)
        adx[2*period-1] = np.mean(dx[period:2*period])
        for i in range(2*period, len(high)):
            adx[i] = (adx[i-1] * (period-1) + dx[i]) / period
        
        return adx
    
    adx_1w = calculate_adx(high_1w, low_1w, close_1w, 14)
    trending = adx_1w >= 25  # Strong trend when ADX >= 25
    
    # Weekly EMA(20) for trend direction
    close_1w_series = pd.Series(close_1w)
    ema_20_1w = close_1w_series.ewm(span=20, adjust=False, min_periods=20).mean().values
    ema_20_prev = np.roll(ema_20_1w, 1)
    ema_20_prev[0] = ema_20_1w[0]
    weekly_uptrend = ema_20_1w > ema_20_prev
    
    # Daily Donchian channels (20-period)
    def calculate_donchian(high, low, period=20):
        upper = np.full_like(high, np.nan)
        lower = np.full_like(high, np.nan)
        for i in range(period-1, len(high)):
            upper[i] = np.max(high[i-period+1:i+1])
            lower[i] = np.min(low[i-period+1:i+1])
        return upper, lower
    
    donchian_upper, donchian_lower = calculate_donchian(high, low, 20)
    
    # Volume confirmation: 20-period volume average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_confirm = volume > (vol_ma * 1.5)
    
    # Align weekly indicators to daily timeframe
    trending_aligned = align_htf_to_ltf(prices, df_1w, trending.astype(float))
    weekly_uptrend_aligned = align_htf_to_ltf(prices, df_1w, weekly_uptrend.astype(float))
    adx_1w_aligned = align_htf_to_ltf(prices, df_1w, adx_1w)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 20  # Donchian period
    
    for i in range(start_idx, n):
        # Skip if any critical data is NaN
        if (np.isnan(donchian_upper[i]) or np.isnan(donchian_lower[i]) or
            np.isnan(trending_aligned[i]) or np.isnan(weekly_uptrend_aligned[i]) or
            np.isnan(vol_ma[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            if trending_aligned[i] > 0.5:  # Trending market
                # Breakout long in uptrend
                if (weekly_uptrend_aligned[i] > 0.5 and
                    close[i] >= donchian_upper[i] and
                    vol_confirm[i]):
                    signals[i] = 0.30
                    position = 1
                # Breakout short in downtrend
                elif (weekly_uptrend_aligned[i] <= 0.5 and
                      close[i] <= donchian_lower[i] and
                      vol_confirm[i]):
                    signals[i] = -0.30
                    position = -1
            else:  # Ranging market (ADX < 25)
                # Fade at upper band (sell high)
                if (close[i] >= donchian_upper[i] and
                    vol_confirm[i]):
                    signals[i] = -0.30
                    position = -1
                # Fade at lower band (buy low)
                elif (close[i] <= donchian_lower[i] and
                      vol_confirm[i]):
                    signals[i] = 0.30
                    position = 1
        elif position == 1:
            # Long exit: reverse signal or stop
            if trending_aligned[i] > 0.5:  # Trending market
                if weekly_uptrend_aligned[i] <= 0.5:  # Trend turned down
                    signals[i] = 0.0
                    position = 0
                elif close[i] <= donchian_lower[i]:  # Break below lower band
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = 0.30
            else:  # Ranging market
                if close[i] <= donchian_lower[i]:  # Hit lower band
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = 0.30
        elif position == -1:
            # Short exit: reverse signal or stop
            if trending_aligned[i] > 0.5:  # Trending market
                if weekly_uptrend_aligned[i] > 0.5:  # Trend turned up
                    signals[i] = 0.0
                    position = 0
                elif close[i] >= donchian_upper[i]:  # Break above upper band
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = -0.30
            else:  # Ranging market
                if close[i] >= donchian_upper[i]:  # Hit upper band
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = -0.30
    
    return signals
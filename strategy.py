#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Hypothesis: 6h Elder Ray + 1d ADX regime filter
    # Elder Ray measures bull/bear power relative to 13-period EMA
    # 1d ADX > 25 indicates trending market (follow Elder Ray signals)
    # 1d ADX < 20 indicates ranging market (fade Elder Ray extremes)
    # Works in bull/bear by adapting to regime: trend follow in trend, mean revert in range
    # Target: 12-37 trades/year per symbol (50-150 over 4 years)
    
    # Session filter: 8:00-20:00 UTC (avoid low volume Asian session)
    hours = pd.DatetimeIndex(prices['open_time']).hour
    in_session = (hours >= 8) & (hours <= 20)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    
    # Get 1d data for ADX regime filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    
    # Calculate 1d ADX (14-period)
    def calculate_adx(high, low, close, period=14):
        # True Range
        tr1 = high[1:] - low[1:]
        tr2 = np.abs(high[1:] - close[:-1])
        tr3 = np.abs(low[1:] - close[:-1])
        tr = np.maximum(tr1, np.maximum(tr2, tr3))
        tr = np.concatenate([[np.nan], tr])  # align with original indices
        
        # Directional Movement
        up_move = high[1:] - high[:-1]
        down_move = low[:-1] - low[1:]
        plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0.0)
        minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0.0)
        plus_dm = np.concatenate([[0.0], plus_dm])
        minus_dm = np.concatenate([[0.0], minus_dm])
        
        # Smoothed TR, +DM, -DM (Wilder's smoothing = EMA with alpha=1/period)
        atr = np.full_like(tr, np.nan)
        plus_dm_smooth = np.full_like(tr, np.nan)
        minus_dm_smooth = np.full_like(tr, np.nan)
        
        # First values: simple average
        if len(tr) >= period:
            atr[period] = np.nanmean(tr[1:period+1])
            plus_dm_smooth[period] = np.nanmean(plus_dm[1:period+1])
            minus_dm_smooth[period] = np.nanmean(minus_dm[1:period+1])
            
            # Wilder's smoothing for subsequent values
            for i in range(period+1, len(tr)):
                atr[i] = (atr[i-1] * (period-1) + tr[i]) / period
                plus_dm_smooth[i] = (plus_dm_smooth[i-1] * (period-1) + plus_dm[i]) / period
                minus_dm_smooth[i] = (minus_dm_smooth[i-1] * (period-1) + minus_dm[i]) / period
        
        # Directional Indicators
        plus_di = np.full_like(tr, np.nan)
        minus_di = np.full_like(tr, np.nan)
        dx = np.full_like(tr, np.nan)
        
        for i in range(period, len(tr)):
            if atr[i] > 0:
                plus_di[i] = 100 * plus_dm_smooth[i] / atr[i]
                minus_di[i] = 100 * minus_dm_smooth[i] / atr[i]
                if plus_di[i] + minus_di[i] > 0:
                    dx[i] = 100 * np.abs(plus_di[i] - minus_di[i]) / (plus_di[i] + minus_di[i])
        
        # ADX: EMA of DX
        adx = np.full_like(tr, np.nan)
        if len(dx) >= 2*period:
            # First ADX value: simple average of first 'period' DX values
            adx[2*period-1] = np.nanmean(dx[period:2*period])
            # Subsequent values: EMA with alpha=1/period
            for i in range(2*period, len(tr)):
                if not np.isnan(dx[i]):
                    adx[i] = (adx[i-1] * (period-1) + dx[i]) / period
        
        return adx
    
    adx_1d = calculate_adx(high_1d, low_1d, close_1d, 14)
    adx_1d_aligned = align_htf_to_ltf(prices, df_1d, adx_1d)
    
    # Calculate 6h Elder Ray (Bull Power and Bear Power)
    # Bull Power = High - EMA(13)
    # Bear Power = Low - EMA(13)
    ema_13 = np.full(n, np.nan)
    if n >= 13:
        # First value: simple average
        ema_13[12] = np.mean(close[:13])
        # Subsequent values: EMA
        for i in range(13, n):
            ema_13[i] = (close[i] * 2/14) + (ema_13[i-1] * 12/14)
    
    bull_power = high - ema_13
    bear_power = low - ema_13
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(100, n):
        if not in_session[i]:
            signals[i] = 0.0
            continue
        
        # Skip if data not ready
        if (np.isnan(adx_1d_aligned[i]) or np.isnan(bull_power[i]) or 
            np.isnan(bear_power[i])):
            signals[i] = 0.0
            continue
        
        # Regime-based logic
        if adx_1d_aligned[i] > 25:  # Trending market - follow Elder Ray
            # Long when bull power is strong and rising
            long_entry = bull_power[i] > 0 and bull_power[i] > bull_power[i-1]
            # Short when bear power is strong and rising (more negative)
            short_entry = bear_power[i] < 0 and bear_power[i] < bear_power[i-1]
            # Exit when power fades
            long_exit = bull_power[i] < 0
            short_exit = bear_power[i] > 0
        elif adx_1d_aligned[i] < 20:  # Ranging market - fade Elder Ray extremes
            # Long when bear power is extremely negative (oversold)
            long_entry = bear_power[i] < -np.std(bear_power[max(0, i-50):i]) * 1.5
            # Short when bull power is extremely positive (overbought)
            short_entry = bull_power[i] > np.std(bull_power[max(0, i-50):i]) * 1.5
            # Exit when power returns to neutral
            long_exit = bear_power[i] > -np.std(bear_power[max(0, i-50):i]) * 0.5
            short_exit = bull_power[i] < np.std(bull_power[max(0, i-50):i]) * 0.5
        else:  # Transition regime - hold or reduce
            long_entry = False
            short_entry = False
            long_exit = position == 1
            short_exit = position == -1
        
        if long_entry and position != 1:
            position = 1
            signals[i] = 0.25
        elif short_entry and position != -1:
            position = -1
            signals[i] = -0.25
        elif position == 1 and long_exit:
            position = 0
            signals[i] = 0.0
        elif position == -1 and short_exit:
            position = 0
            signals[i] = 0.0
        else:
            # Hold current position
            if position == 1:
                signals[i] = 0.25
            elif position == -1:
                signals[i] = -0.25
            else:
                signals[i] = 0.0
    
    return signals

name = "6h_1d_elder_ray_adx_regime_v1"
timeframe = "6h"
leverage = 1.0
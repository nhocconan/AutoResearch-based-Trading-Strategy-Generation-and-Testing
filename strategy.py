#!/usr/bin/env python3
# Hypothesis: 1d Donchian(20) breakout with 1w EMA34 trend filter, volume confirmation (>1.5x 20-period average), and ADX regime filter (ADX > 25 for trending, < 20 for ranging). Uses discrete position sizing (0.0, ±0.25) to minimize fee churn. Designed to work in both bull and bear markets by combining price structure (Donchian), trend (1w EMA34), volume strength, and regime awareness.

name = "1d_Donchian20_Breakout_1wEMA34_VolumeADX_v1"
timeframe = "1d"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    open_ = prices['open'].values
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # --- 1d Indicators (LTF) ---
    # Volume confirmation: > 1.5x 20-period average (moderate threshold to reduce trades)
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_confirm = volume > (1.5 * vol_ma_20)
    
    # ADX (14) - regime filter
    def calculate_adx(high, low, close, period=14):
        plus_dm = np.zeros_like(high)
        minus_dm = np.zeros_like(high)
        tr = np.zeros_like(high)
        
        for i in range(1, len(high)):
            plus_dm[i] = max(high[i] - high[i-1], 0) if (high[i] - high[i-1]) > (low[i-1] - low[i]) else 0
            minus_dm[i] = max(low[i-1] - low[i], 0) if (low[i-1] - low[i]) > (high[i] - high[i-1]) else 0
            tr[i] = max(high[i] - low[i], abs(high[i] - close[i-1]), abs(low[i] - close[i-1]))
        
        # Wilder's smoothing
        atr = np.zeros_like(high)
        plus_dm_smooth = np.zeros_like(high)
        minus_dm_smooth = np.zeros_like(high)
        
        atr[period] = np.mean(tr[1:period+1])
        plus_dm_smooth[period] = np.mean(plus_dm[1:period+1])
        minus_dm_smooth[period] = np.mean(minus_dm[1:period+1])
        
        for i in range(period+1, len(high)):
            atr[i] = (atr[i-1] * (period-1) + tr[i]) / period
            plus_dm_smooth[i] = (plus_dm_smooth[i-1] * (period-1) + plus_dm[i]) / period
            minus_dm_smooth[i] = (minus_dm_smooth[i-1] * (period-1) + minus_dm[i]) / period
        
        plus_di = 100 * plus_dm_smooth / atr
        minus_di = 100 * minus_dm_smooth / atr
        dx = np.zeros_like(high)
        dx = 100 * np.abs(plus_di - minus_di) / (plus_di + minus_di)
        adx = np.zeros_like(high)
        adx[2*period] = np.mean(dx[period+1:2*period+1])
        for i in range(2*period+1, len(high)):
            adx[i] = (adx[i-1] * (period-1) + dx[i]) / period
        
        return adx
    
    adx = calculate_adx(high, low, close, 14)
    adx_trending = adx > 25   # trending market (breakout follow)
    adx_ranging = adx < 20    # ranging market (mean reversion)
    
    # --- 1w Indicators (HTF) ---
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 50:
        return np.zeros(n)
    close_1w = df_1w['close'].values
    
    # 1w EMA(34) - trend filter
    ema_34_1w = pd.Series(close_1w).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_34_1w)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(1, n):
        # Skip if missing data
        if (np.isnan(ema_34_1w_aligned[i]) or
            np.isnan(volume_confirm[i]) or
            np.isnan(adx[i])):
            signals[i] = 0.0
            continue
        
        # Calculate Donchian levels for today (using previous 20 days)
        if i >= 20:
            donchian_high = np.max(high[i-20:i])
            donchian_low = np.min(low[i-20:i])
        else:
            donchian_high = np.nan
            donchian_low = np.nan
        
        if position == 0:
            # LONG: Price breaks above Donchian high AND close > 1w EMA34 (bullish trend) AND volume confirm AND (ranging: mean reversion OR trending: momentum)
            if (not np.isnan(donchian_high) and 
                close[i] > donchian_high and 
                close[i] > ema_34_1w_aligned[i] and 
                volume_confirm[i] and
                (adx_ranging[i] or adx_trending[i])):  # allow both regimes but with different logic implicit in entry
                signals[i] = 0.25
                position = 1
            # SHORT: Price breaks below Donchian low AND close < 1w EMA34 (bearish trend) AND volume confirm AND (ranging: mean reversion OR trending: momentum)
            elif (not np.isnan(donchian_low) and 
                  close[i] < donchian_low and 
                  close[i] < ema_34_1w_aligned[i] and 
                  volume_confirm[i] and
                  (adx_ranging[i] or adx_trending[i])):
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: Price crosses below Donchian low (breakdown) OR touches Donchian high (mean reversion in ranging)
            if close[i] < donchian_low or (adx_ranging[i] and close[i] < donchian_high):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: Price crosses above Donchian high (breakout) OR touches Donchian low (mean reversion in ranging)
            if close[i] > donchian_high or (adx_ranging[i] and close[i] > donchian_low):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals
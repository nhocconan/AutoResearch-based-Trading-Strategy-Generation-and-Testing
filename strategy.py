#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Elder Ray power with 12h regime filter and volume confirmation
# Elder Ray: Bull Power = High - EMA(13), Bear Power = EMA(13) - Low
# Regime: 12h ADX > 25 = trending (use Elder Ray), ADX < 20 = ranging (mean reversion at Bollinger Bands)
# Volume: Current volume > 20-period average for confirmation
# Target: 50-150 total trades over 4 years with balanced long/short in bull and bear markets
# Uses 6h timeframe with 12h regime filter to adapt to market conditions

name = "6h_elderray_12h_regime_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Price data
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # 12h data for regime filter
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 30:
        return np.zeros(n)
    
    high_12h = df_12h['high'].values
    low_12h = df_12h['low'].values
    close_12h = df_12h['close'].values
    
    # 12h ADX for regime detection (trending vs ranging)
    def calculate_adx(high, low, close, period=14):
        plus_dm = np.zeros_like(high)
        minus_dm = np.zeros_like(high)
        tr = np.zeros_like(high)
        
        for i in range(1, len(high)):
            plus_dm[i] = max(0, high[i] - high[i-1])
            minus_dm[i] = max(0, low[i-1] - low[i])
            if plus_dm[i] == minus_dm[i]:
                plus_dm[i] = 0
                minus_dm[i] = 0
            tr[i] = max(high[i] - low[i], abs(high[i] - close[i-1]), abs(low[i] - close[i-1]))
        
        # Smoothed values
        atr = np.zeros_like(high)
        plus_dm_smooth = np.zeros_like(high)
        minus_dm_smooth = np.zeros_like(high)
        
        atr[period-1] = np.mean(tr[1:period+1])
        plus_dm_smooth[period-1] = np.mean(plus_dm[1:period+1])
        minus_dm_smooth[period-1] = np.mean(minus_dm[1:period+1])
        
        for i in range(period, len(high)):
            atr[i] = (atr[i-1] * (period-1) + tr[i]) / period
            plus_dm_smooth[i] = (plus_dm_smooth[i-1] * (period-1) + plus_dm[i]) / period
            minus_dm_smooth[i] = (minus_dm_smooth[i-1] * (period-1) + minus_dm[i]) / period
        
        plus_di = 100 * plus_dm_smooth / atr
        minus_di = 100 * minus_dm_smooth / atr
        dx = np.zeros_like(high)
        dx[:] = np.where((plus_di + minus_di) != 0, 100 * np.abs(plus_di - minus_di) / (plus_di + minus_di), 0)
        
        adx = np.zeros_like(high)
        adx[2*period-1] = np.mean(dx[period:2*period+1])
        for i in range(2*period, len(high)):
            adx[i] = (adx[i-1] * (period-1) + dx[i]) / period
        
        return adx
    
    adx_12h = calculate_adx(high_12h, low_12h, close_12h, 14)
    adx_12h_aligned = align_htf_to_ltf(prices, df_12h, adx_12h)
    
    # Elder Ray components (13-period EMA)
    ema_13 = pd.Series(close).ewm(span=13, min_periods=13, adjust=False).mean().values
    bull_power = high - ema_13
    bear_power = ema_13 - low
    
    # Bollinger Bands for ranging regime (20-period)
    bb_middle = pd.Series(close).rolling(window=20, min_periods=20).mean().values
    bb_std = pd.Series(close).rolling(window=20, min_periods=20).std().values
    bb_upper = bb_middle + 2 * bb_std
    bb_lower = bb_middle - 2 * bb_std
    
    # Volume filter
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_filter = volume > vol_ma
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    for i in range(30, n):
        # Skip if required data not available
        if (np.isnan(ema_13[i]) or np.isnan(bull_power[i]) or np.isnan(bear_power[i]) or
            np.isnan(adx_12h_aligned[i]) or np.isnan(bb_upper[i]) or np.isnan(bb_lower[i]) or
            np.isnan(vol_ma[i])):
            if position != 0:
                signals[i] = position * 0.25
            else:
                signals[i] = 0.0
            continue
        
        if position == 1:  # long position
            # Stoploss: 2 * ATR approximation
            if close[i] < entry_price - 2.0 * (high[i] - low[i]):
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
            # Exit conditions based on regime
            elif adx_12h_aligned[i] > 25:  # trending regime
                if bear_power[i] > 0 or ema_13[i] < ema_13[i-1]:  # bear power positive or EMA falling
                    signals[i] = 0.0
                    position = 0
                    entry_price = 0.0
            else:  # ranging regime
                if close[i] > bb_middle[i]:  # mean reversion to middle
                    signals[i] = 0.0
                    position = 0
                    entry_price = 0.0
            else:
                signals[i] = 0.25
        elif position == -1:  # short position
            # Stoploss: 2 * ATR approximation
            if close[i] > entry_price + 2.0 * (high[i] - low[i]):
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
            # Exit conditions based on regime
            elif adx_12h_aligned[i] > 25:  # trending regime
                if bull_power[i] < 0 or ema_13[i] > ema_13[i-1]:  # bull power negative or EMA rising
                    signals[i] = 0.0
                    position = 0
                    entry_price = 0.0
            else:  # ranging regime
                if close[i] < bb_middle[i]:  # mean reversion to middle
                    signals[i] = 0.0
                    position = 0
                    entry_price = 0.0
            else:
                signals[i] = -0.25
        else:
            # Look for entries with volume confirmation
            if vol_filter[i]:
                if adx_12h_aligned[i] > 25:  # trending regime - follow Elder Ray
                    # Long when bull power positive and rising
                    if bull_power[i] > 0 and bull_power[i] > bull_power[i-1]:
                        signals[i] = 0.25
                        position = 1
                        entry_price = close[i]
                    # Short when bear power positive and rising
                    elif bear_power[i] > 0 and bear_power[i] > bear_power[i-1]:
                        signals[i] = -0.25
                        position = -1
                        entry_price = close[i]
                else:  # ranging regime - mean reversion at Bollinger Bands
                    # Long when price touches lower band and bull power turning positive
                    if close[i] <= bb_lower[i] and bull_power[i] > 0 and bull_power[i] > bull_power[i-1]:
                        signals[i] = 0.25
                        position = 1
                        entry_price = close[i]
                    # Short when price touches upper band and bear power turning positive
                    elif close[i] >= bb_upper[i] and bear_power[i] > 0 and bear_power[i] > bear_power[i-1]:
                        signals[i] = -0.25
                        position = -1
                        entry_price = close[i]
    
    return signals
#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # === 4h data (HTF for trend and structure) ===
    df_4h = get_htf_data(prices, '4h')
    close_4h = df_4h['close'].values
    high_4h = df_4h['high'].values
    low_4h = df_4h['low'].values
    volume_4h = df_4h['volume'].values
    
    # === 1d data (HTF for regime filter) ===
    df_1d = get_htf_data(prices, '1d')
    close_1d = df_1d['close'].values
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    
    # === 4h EMA50 (trend filter) ===
    ema_50_4h = pd.Series(close_4h).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_4h_aligned = align_htf_to_ltf(prices, df_4h, ema_50_4h)
    
    # === 4h ATR(14) for volatility filter ===
    tr1 = high_4h - low_4h
    tr2 = np.abs(high_4h - np.roll(close_4h, 1))
    tr3 = np.abs(low_4h - np.roll(close_4h, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = 0
    atr_14_4h = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    atr_14_4h_aligned = align_htf_to_ltf(prices, df_4h, atr_14_4h)
    
    # === 4h Donchian channel (20-period) ===
    donch_high = pd.Series(high_4h).rolling(window=20, min_periods=20).max().values
    donch_low = pd.Series(low_4h).rolling(window=20, min_periods=20).min().values
    
    # Shift by 1 to avoid look-ahead (use previous bar's channel)
    donch_high = np.roll(donch_high, 1)
    donch_low = np.roll(donch_low, 1)
    donch_high[0] = np.nan
    donch_low[0] = np.nan
    
    # === 4h volume ratio for confirmation ===
    vol_ma_15_4h = pd.Series(volume_4h).rolling(window=15, min_periods=15).mean().values
    vol_ratio_4h = volume_4h / vol_ma_15_4h
    
    # === 1d ADX(14) for regime filter (trending vs ranging) ===
    # Calculate +DM, -DM, TR
    up_move = high_1d[1:] - high_1d[:-1]
    down_move = low_1d[:-1] - low_1d[1:]
    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0)
    tr_dm = np.maximum(tr1[1:], np.maximum(np.abs(high_1d[1:] - close_1d[:-1]), np.abs(low_1d[1:] - close_1d[:-1])))
    tr_dm = np.concatenate([[0], tr_dm])
    
    # Smooth with Wilder's smoothing (alpha = 1/period)
    def wilder_smooth(data, period):
        result = np.zeros_like(data)
        alpha = 1.0 / period
        result[period-1] = np.mean(data[:period])
        for i in range(period, len(data)):
            result[i] = alpha * data[i] + (1 - alpha) * result[i-1]
        return result
    
    atr_14_dm = wilder_smooth(tr_dm, 14)
    plus_di_14 = 100 * wilder_smooth(plus_dm, 14) / atr_14_dm
    minus_di_14 = 100 * wilder_smooth(minus_dm, 14) / atr_14_dm
    dx = 100 * np.abs(plus_di_14 - minus_di_14) / (plus_di_14 + minus_di_14 + 1e-10)
    adx_14 = wilder_smooth(dx, 14)
    adx_14_aligned = align_htf_to_ltf(prices, df_1d, adx_14)
    
    # === 1h session filter (08-20 UTC) ===
    hours = pd.DatetimeIndex(prices['open_time']).hour
    
    signals = np.zeros(n)
    
    # Warmup
    warmup = 100
    
    # Track position and entry price for stoploss
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    for i in range(warmup, n):
        # Session filter: only trade 08-20 UTC
        hour = hours[i]
        if hour < 8 or hour > 20:
            if position != 0:
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
            else:
                signals[i] = 0.0
            continue
        
        # Skip if any data is NaN
        if (np.isnan(donch_high[i]) or 
            np.isnan(donch_low[i]) or 
            np.isnan(ema_50_4h_aligned[i]) or 
            np.isnan(atr_14_4h_aligned[i]) or
            np.isnan(vol_ratio_4h[i]) or
            np.isnan(adx_14_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
            else:
                signals[i] = 0.0
            continue
        
        price = close[i]
        upper = donch_high[i]
        lower = donch_low[i]
        ema_trend = ema_50_4h_aligned[i]
        atr = atr_14_4h_aligned[i]
        vol_ratio = vol_ratio_4h[i]
        adx = adx_14_aligned[i]
        
        # === STOPLOSS LOGIC ===
        if position == 1:  # Long position
            # Stop loss: price closes below entry - 2.0 * ATR
            if price < entry_price - 2.0 * atr:
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
                continue
        
        elif position == -1:  # Short position
            # Stop loss: price closes above entry + 2.0 * ATR
            if price > entry_price + 2.0 * atr:
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
                continue
        
        # === EXIT LOGIC ===
        if position == 1:  # Long position
            # Exit: price closes below Donchian lower OR trend reverses (below EMA50)
            if price < lower or price < ema_trend:
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
                continue
        
        elif position == -1:  # Short position
            # Exit: price closes above Donchian upper OR trend reverses (above EMA50)
            if price > upper or price > ema_trend:
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
                continue
        
        # === ENTRY LOGIC (only when flat) ===
        if position == 0:
            # Only trade in trending markets (ADX > 25)
            if adx > 25:
                # LONG: Break above Donchian upper with volume, in uptrend (above EMA50)
                if (price > upper and vol_ratio > 1.8 and price > ema_trend):
                    signals[i] = 0.20
                    position = 1
                    entry_price = price
                    continue
                # SHORT: Break below Donchian lower with volume, in downtrend (below EMA50)
                elif (price < lower and vol_ratio > 1.8 and price < ema_trend):
                    signals[i] = -0.20
                    position = -1
                    entry_price = price
                    continue
        
        # Hold current position
        if position == 1:
            signals[i] = 0.20
        elif position == -1:
            signals[i] = -0.20
        else:
            signals[i] = 0.0
    
    return signals

name = "1h_4hDonchian_ADX25_VolumeFilter"
timeframe = "1h"
leverage = 1.0
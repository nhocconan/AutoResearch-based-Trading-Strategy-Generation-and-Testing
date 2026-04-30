#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Donchian(20) breakout with volume confirmation and 1d ADX regime filter
# Long when price breaks above 20-period Donchian high + volume spike + 1d ADX > 25 (trending)
# Short when price breaks below 20-period Donchian low + volume spike + 1d ADX > 25 (trending)
# Uses discrete sizing 0.25 to balance profit and fee drag. Target: 75-200 total trades over 4 years (19-50/year).
# Donchian provides clear structure, volume confirms institutional interest, ADX filters ranging markets.
# Works in both bull and bear markets by only trading in trending regimes (ADX > 25).

name = "4h_Donchian20_VolumeSpike_1dADX25_Trend_v1"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    open_time = prices['open_time'].values
    
    # Pre-compute session hours (08-20 UTC)
    hours = pd.DatetimeIndex(open_time).hour
    in_session = (hours >= 8) & (hours <= 20)
    
    # Calculate 4h Donchian channels (20-period)
    donchian_h = pd.Series(high).rolling(window=20, min_periods=20).max().values
    donchian_l = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # Volume confirmation: volume > 2.0x 24-period average (strict to reduce trades)
    vol_ma_24 = pd.Series(volume).rolling(window=24, min_periods=24).mean().values
    volume_spike = volume > (2.0 * vol_ma_24)
    
    # Calculate 1d ADX(14) for trend filter (HTF)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 14:
        return np.zeros(n)
    
    # True Range calculation
    tr1 = df_1d['high'][1:].values - df_1d['low'][1:].values
    tr2 = np.abs(df_1d['high'][1:].values - df_1d['close'][:-1].values)
    tr3 = np.abs(df_1d['low'][1:].values - df_1d['close'][:-1].values)
    tr_1d = np.concatenate([[np.nan], np.maximum(tr1, np.maximum(tr2, tr3))])
    
    # Directional Movement
    dm_plus = np.where((df_1d['high'][1:].values - df_1d['high'][:-1].values) > 
                       (df_1d['low'][:-1].values - df_1d['low'][1:].values),
                       np.maximum(df_1d['high'][1:].values - df_1d['high'][:-1].values, 0), 0)
    dm_minus = np.where((df_1d['low'][:-1].values - df_1d['low'][1:].values) > 
                        (df_1d['high'][1:].values - df_1d['high'][:-1].values),
                        np.maximum(df_1d['low'][:-1].values - df_1d['low'][1:].values, 0), 0)
    
    # Smooth TR, DM+ and DM- with Wilder's smoothing (EMA with alpha=1/period)
    def wilder_smooth(data, period):
        alpha = 1.0 / period
        result = np.full_like(data, np.nan)
        result[period-1] = np.nanmean(data[:period])  # seed with simple average
        for i in range(period, len(data)):
            result[i] = alpha * data[i] + (1 - alpha) * result[i-1]
        return result
    
    tr_smooth = wilder_smooth(tr_1d, 14)
    dm_plus_smooth = wilder_smooth(dm_plus, 14)
    dm_minus_smooth = wilder_smooth(dm_minus, 14)
    
    # DI+ and DI-
    di_plus = 100 * dm_plus_smooth / tr_smooth
    di_minus = 100 * dm_minus_smooth / tr_smooth
    
    # DX and ADX
    dx = 100 * np.abs(di_plus - di_minus) / (di_plus + di_minus)
    adx = wilder_smooth(dx, 14)
    
    # Align 1d ADX to 4h timeframe
    adx_1d_aligned = align_htf_to_ltf(prices, df_1d, adx)
    
    # ATR for stoploss (14-period)
    tr1 = high[1:] - low[1:]
    tr2 = np.abs(high[1:] - close[:-1])
    tr3 = np.abs(low[1:] - close[:-1])
    tr = np.concatenate([[np.nan], np.maximum(tr1, np.maximum(tr2, tr3))])
    atr_14 = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    start_idx = max(100, 24, 20, 14)  # warmup
    
    for i in range(start_idx, n):
        # Skip if indicators not ready or outside session
        if (np.isnan(donchian_h[i]) or np.isnan(donchian_l[i]) or
            np.isnan(vol_ma_24[i]) or np.isnan(adx_1d_aligned[i]) or
            np.isnan(atr_14[i]) or not in_session[i]):
            signals[i] = 0.0
            continue
            
        curr_close = close[i]
        curr_high = high[i]
        curr_low = low[i]
        curr_donchian_h = donchian_h[i]
        curr_donchian_l = donchian_l[i]
        curr_volume_spike = volume_spike[i]
        curr_adx = adx_1d_aligned[i]
        curr_atr = atr_14[i]
        
        if position == 0:  # Flat - look for new entries
            # Only trade on volume spike with Donchian break and 1d ADX > 25 (trending)
            if curr_volume_spike and curr_adx > 25:
                # Bullish: Close breaks above Donchian high
                if curr_close > curr_donchian_h:
                    signals[i] = 0.25
                    position = 1
                    entry_price = curr_close
                # Bearish: Close breaks below Donchian low
                elif curr_close < curr_donchian_l:
                    signals[i] = -0.25
                    position = -1
                    entry_price = curr_close
        
        elif position == 1:  # Long position
            # Stoploss: 2 * ATR below entry
            stop_loss = entry_price - 2.0 * curr_atr
            # Exit: Stoploss hit OR close drops below Donchian low OR ADX < 20 (ranging)
            if curr_low <= stop_loss or curr_close < curr_donchian_l or curr_adx < 20:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:  # Short position
            # Stoploss: 2 * ATR above entry
            stop_loss = entry_price + 2.0 * curr_atr
            # Exit: Stoploss hit OR close rises above Donchian high OR ADX < 20 (ranging)
            if curr_high >= stop_loss or curr_close > curr_donchian_h or curr_adx < 20:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals
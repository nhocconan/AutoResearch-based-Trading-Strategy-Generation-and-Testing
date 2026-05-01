#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Camarilla R3/S3 breakout with 12h ADX trend filter and volume spike confirmation.
# Long when price breaks above Camarilla R3 level AND 12h ADX > 25 AND volume > 2.0x 20-period volume median.
# Short when price breaks below Camarilla S3 level AND 12h ADX > 25 AND volume > 2.0x 20-period volume median.
# Uses discrete sizing 0.25. ATR(14) stoploss: signal→0 when price moves against position by 2.0*ATR.
# Camarilla R3/S3 levels provide strong support/resistance with fewer false breakouts than R4/S4.
# 12h ADX > 25 filters for strong trending markets only, reducing whipsaws in ranging conditions.
# Volume spike confirms breakout conviction. Works in bull markets (breakouts with trend) and bear markets (breakdowns with trend).
# Target: 19-50 trades/year on 4h timeframe (75-200 total over 4 years).

name = "4h_Camarilla_R3S3_Breakout_12hADX_Volume_v1"
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
    
    # Calculate ATR(14) for stoploss
    tr1 = high[1:] - low[1:]
    tr2 = np.abs(high[1:] - close[:-1])
    tr3 = np.abs(low[1:] - close[:-1])
    tr_first = np.max([high[0] - low[0], np.abs(high[0] - close[0]), np.abs(low[0] - close[0])])
    tr = np.concatenate([[tr_first], np.maximum(tr1, np.maximum(tr2, tr3))])
    atr = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    # Calculate 20-period volume median for volume confirmation
    vol_median_20 = pd.Series(volume).rolling(window=20, min_periods=20).median().values
    
    # Calculate Camarilla levels (using prior bar's OHLC to avoid look-ahead)
    prev_close = np.concatenate([[close[0]], close[:-1]])
    prev_high = np.concatenate([[high[0]], high[:-1]])
    prev_low = np.concatenate([[low[0]], low[:-1]])
    camarilla_range = prev_high - prev_low
    camarilla_R3 = prev_close + 1.25 * camarilla_range
    camarilla_S3 = prev_close - 1.25 * camarilla_range
    
    # Calculate 12h ADX trend filter (HTF)
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 14:
        return np.zeros(n)
    
    # ADX calculation: +DI, -DI, DX, then ADX
    high_12h = df_12h['high'].values
    low_12h = df_12h['low'].values
    close_12h = df_12h['close'].values
    
    # True Range
    tr_12h1 = high_12h[1:] - low_12h[1:]
    tr_12h2 = np.abs(high_12h[1:] - close_12h[:-1])
    tr_12h3 = np.abs(low_12h[1:] - close_12h[:-1])
    tr_12h_first = np.max([high_12h[0] - low_12h[0], np.abs(high_12h[0] - close_12h[0]), np.abs(low_12h[0] - close_12h[0])])
    tr_12h = np.concatenate([[tr_12h_first], np.maximum(tr_12h1, np.maximum(tr_12h2, tr_12h3))])
    
    # +DM and -DM
    up_move = high_12h[1:] - high_12h[:-1]
    down_move = low_12h[:-1] - low_12h[1:]
    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0.0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0.0)
    
    # Smooth TR, +DM, -DM with Wilder's smoothing (alpha = 1/period)
    def wilders_smoothing(values, period):
        if len(values) < period:
            return np.full_like(values, np.nan)
        result = np.full_like(values, np.nan)
        # First value is simple average
        result[period-1] = np.mean(values[:period])
        # Subsequent values: Wilder's smoothing
        for i in range(period, len(values)):
            result[i] = (result[i-1] * (period-1) + values[i]) / period
        return result
    
    atr_12h = wilders_smoothing(tr_12h, 14)
    plus_dm_smooth = wilders_smoothing(plus_dm, 14)
    minus_dm_smooth = wilders_smoothing(minus_dm, 14)
    
    # +DI and -DI
    plus_di_12h = 100 * plus_dm_smooth / atr_12h
    minus_di_12h = 100 * minus_dm_smooth / atr_12h
    
    # DX
    dx_12h = np.where((plus_di_12h + minus_di_12h) != 0, 
                      100 * np.abs(plus_di_12h - minus_di_12h) / (plus_di_12h + minus_di_12h), 
                      0.0)
    
    # ADX: smoothed DX
    adx_12h = wilders_smoothing(dx_12h, 14)
    adx_12h_aligned = align_htf_to_ltf(prices, df_12h, adx_12h)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0  # track entry price for stoploss
    
    # Start after warmup for ATR, ADX, volume, and Camarilla
    start_idx = 100
    
    for i in range(start_idx, n):
        if (np.isnan(atr[i]) or 
            np.isnan(adx_12h_aligned[i]) or 
            np.isnan(camarilla_R3[i]) or 
            np.isnan(camarilla_S3[i]) or 
            np.isnan(vol_median_20[i])):
            signals[i] = 0.0
            continue
        
        curr_close = close[i]
        curr_volume = volume[i]
        
        # Trend filter: 12h ADX > 25 indicates strong trend
        strong_trend = adx_12h_aligned[i] > 25
        
        # Volume confirmation: current volume > 2.0x 20-period volume median
        if vol_median_20[i] <= 0 or np.isnan(vol_median_20[i]):
            volume_confirm = False
        else:
            volume_confirm = curr_volume > (vol_median_20[i] * 2.0)
        
        if position == 0:  # Flat - look for new entries
            # Long: price > Camarilla R3 AND strong trend AND volume spike
            if curr_close > camarilla_R3[i] and strong_trend and volume_confirm:
                signals[i] = 0.25
                position = 1
                entry_price = curr_close
            # Short: price < Camarilla S3 AND strong trend AND volume spike
            elif curr_close < camarilla_S3[i] and strong_trend and volume_confirm:
                signals[i] = -0.25
                position = -1
                entry_price = curr_close
            else:
                signals[i] = 0.0
        
        elif position == 1:  # Long position
            # Stoploss: price moves against position by 2.0*ATR
            if curr_close < entry_price - 2.0 * atr[i]:
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
            # Exit: price breaks below Camarilla S3 OR trend weakens (ADX < 20)
            elif curr_close < camarilla_S3[i] or adx_12h_aligned[i] < 20:
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
            else:
                signals[i] = 0.25
        
        elif position == -1:  # Short position
            # Stoploss: price moves against position by 2.0*ATR
            if curr_close > entry_price + 2.0 * atr[i]:
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
            # Exit: price breaks above Camarilla R3 OR trend weakens (ADX < 20)
            elif curr_close > camarilla_R3[i] or adx_12h_aligned[i] < 20:
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
            else:
                signals[i] = -0.25
    
    return signals
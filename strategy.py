#!/usr/bin/env python3
"""
Hypothesis: 1h timeframe with 4h trend filter and 1d regime filter for BTC/ETH.
Long when: price > 4h EMA50, 1d ADX < 25 (range regime), and RSI(14) crosses above 30 (mean reversion long).
Short when: price < 4h EMA50, 1d ADX < 25 (range regime), and RSI(14) crosses below 70 (mean reversion short).
Exit when RSI crosses 50 (middle) or adverse 4h EMA50 crossover.
Uses session filter (08-20 UTC) and discrete position sizing (0.20) to minimize fee churn.
Targets 15-37 trades/year per symbol by combining tight range mean reversion with HTF trend and regime filters.
The 4h EMA50 provides trend bias while 1d ADX < 25 ensures we only trade in range-bound markets where mean reversion works.
"""

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
    
    # Precompute session hours (08-20 UTC) - avoid look-ahead
    hours = pd.DatetimeIndex(prices["open_time"]).hour
    in_session = (hours >= 8) & (hours <= 20)
    
    # Load 4h data for EMA50 trend filter - ONCE before loop
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 50:
        return np.zeros(n)
    
    close_4h = df_4h['close'].values
    
    # Calculate EMA50 on 4h data
    ema50_4h = pd.Series(close_4h).ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Align 4h EMA50 to 1h timeframe
    ema50_4h_aligned = align_htf_to_ltf(prices, df_4h, ema50_4h)
    
    # Load 1d data for ADX regime filter - ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate ADX(14) on 1d data
    # True Range
    tr1 = np.abs(high_1d[1:] - low_1d[1:])
    tr2 = np.abs(high_1d[1:] - close_1d[:-1])
    tr3 = np.abs(low_1d[1:] - close_1d[:-1])
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr = np.concatenate([[np.nan], tr])  # First value NaN
    
    # Directional Movement
    dm_plus = np.where((high_1d[1:] - high_1d[:-1]) > (low_1d[:-1] - low_1d[1:]), 
                       np.maximum(high_1d[1:] - high_1d[:-1], 0), 0)
    dm_minus = np.where((low_1d[:-1] - low_1d[1:]) > (high_1d[1:] - high_1d[:-1]), 
                        np.maximum(low_1d[:-1] - low_1d[1:], 0), 0)
    dm_plus = np.concatenate([[np.nan], dm_plus])
    dm_minus = np.concatenate([[np.nan], dm_minus])
    
    # Smooth TR, DM+ , DM- with Wilder's smoothing (alpha = 1/14)
    def wilders_smoothing(data, period):
        result = np.full_like(data, np.nan)
        if len(data) >= period:
            # First value is simple average
            result[period-1] = np.nanmean(data[period-1:2*period-1]) if 2*period-1 <= len(data) else np.nanmean(data[period-1:])
            # Rest is Wilder smoothing
            for i in range(period, len(data)):
                if not np.isnan(result[i-1]) and not np.isnan(data[i]):
                    result[i] = (result[i-1] * (period-1) + data[i]) / period
        return result
    
    tr_smoothed = wilders_smoothing(tr, 14)
    dm_plus_smoothed = wilders_smoothing(dm_plus, 14)
    dm_minus_smoothed = wilders_smoothing(dm_minus, 14)
    
    # DI+ and DI-
    di_plus = np.where(tr_smoothed != 0, dm_plus_smoothed / tr_smoothed * 100, 0)
    di_minus = np.where(tr_smoothed != 0, dm_minus_smoothed / tr_smoothed * 100, 0)
    
    # DX and ADX
    dx = np.where((di_plus + di_minus) != 0, np.abs(di_plus - di_minus) / (di_plus + di_minus) * 100, 0)
    adx = wilders_smoothing(dx, 14)
    
    # Align 1d ADX to 1h timeframe with extra delay (ADX needs confirmation)
    adx_aligned = align_htf_to_ltf(prices, df_1d, adx, additional_delay_bars=2)
    
    # Calculate RSI(14) on 1h data
    def rsi(close_prices, period=14):
        delta = np.diff(close_prices)
        delta = np.concatenate([[np.nan], delta])
        gain = np.where(delta > 0, delta, 0)
        loss = np.where(delta < 0, -delta, 0)
        
        # Wilder's smoothing
        avg_gain = np.full_like(gain, np.nan)
        avg_loss = np.full_like(loss, np.nan)
        
        if len(gain) >= period:
            avg_gain[period-1] = np.nanmean(gain[period-1:2*period-1]) if 2*period-1 <= len(gain) else np.nanmean(gain[period-1:])
            avg_loss[period-1] = np.nanmean(loss[period-1:2*period-1]) if 2*period-1 <= len(loss) else np.nanmean(loss[period-1:])
            
            for i in range(period, len(gain)):
                if not np.isnan(avg_gain[i-1]) and not np.isnan(gain[i]):
                    avg_gain[i] = (avg_gain[i-1] * (period-1) + gain[i]) / period
                if not np.isnan(avg_loss[i-1]) and not np.isnan(loss[i]):
                    avg_loss[i] = (avg_loss[i-1] * (period-1) + loss[i]) / period
        
        rs = np.where(avg_loss != 0, avg_gain / avg_loss, 0)
        rsi = 100 - (100 / (1 + rs))
        return rsi
    
    rsi_values = rsi(close, 14)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(100, n):
        # Skip if not in trading session
        if not in_session[i]:
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
            
        # Skip if data not ready
        if (np.isnan(ema50_4h_aligned[i]) or np.isnan(adx_aligned[i]) or 
            np.isnan(rsi_values[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        price = close[i]
        rsi_now = rsi_values[i]
        rsi_prev = rsi_values[i-1]
        
        if position == 0:
            # Long: price > 4h EMA50, range regime (ADX < 25), RSI crosses above 30
            if (price > ema50_4h_aligned[i] and 
                adx_aligned[i] < 25 and 
                rsi_prev <= 30 and rsi_now > 30):
                signals[i] = 0.20
                position = 1
            # Short: price < 4h EMA50, range regime (ADX < 25), RSI crosses below 70
            elif (price < ema50_4h_aligned[i] and 
                  adx_aligned[i] < 25 and 
                  rsi_prev >= 70 and rsi_now < 70):
                signals[i] = -0.20
                position = -1
        else:
            # Exit conditions
            exit_signal = False
            
            if position == 1:
                # Exit long: RSI crosses below 50 OR price < 4h EMA50 (trend change)
                if (rsi_prev >= 50 and rsi_now < 50) or price < ema50_4h_aligned[i]:
                    exit_signal = True
            else:  # position == -1
                # Exit short: RSI crosses above 50 OR price > 4h EMA50 (trend change)
                if (rsi_prev <= 50 and rsi_now > 50) or price > ema50_4h_aligned[i]:
                    exit_signal = True
            
            if exit_signal:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.20 if position == 1 else -0.20
    
    return signals

name = "1H_4hEMA50_1dADX_RSI_MeanReversion_Session"
timeframe = "1h"
leverage = 1.0
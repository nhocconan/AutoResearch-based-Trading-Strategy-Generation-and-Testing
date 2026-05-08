#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "6h_ADX_Trend_Momentum_Confirmation_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 200:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get daily data for trend and momentum filters
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # Calculate ADX on daily timeframe for trend strength
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate True Range (TR)
    tr1 = high_1d[1:] - low_1d[1:]
    tr2 = np.abs(high_1d[1:] - close_1d[:-1])
    tr3 = np.abs(low_1d[1:] - close_1d[:-1])
    tr = np.concatenate([[np.nan], np.maximum(tr1, np.maximum(tr2, tr3))])
    
    # Calculate Directional Movement (DM)
    dm_plus = np.where((high_1d[1:] - high_1d[:-1]) > (low_1d[:-1] - low_1d[1:]), 
                       np.maximum(high_1d[1:] - high_1d[:-1], 0), 0)
    dm_minus = np.where((low_1d[:-1] - low_1d[1:]) > (high_1d[1:] - high_1d[:-1]), 
                        np.maximum(low_1d[:-1] - low_1d[1:], 0), 0)
    dm_plus = np.concatenate([[0], dm_plus])
    dm_minus = np.concatenate([[0], dm_minus])
    
    # Smooth TR and DM using Wilder's smoothing (equivalent to EMA with alpha=1/period)
    atr_period = 14
    tr_sum = np.nancumsum(tr) - np.concatenate([[0], np.nancumsum(tr)[:-atr_period]]) if len(tr) >= atr_period else np.full_like(tr, np.nan)
    dm_plus_sum = np.nancumsum(dm_plus) - np.concatenate([[0], np.nancumsum(dm_plus)[:-atr_period]]) if len(dm_plus) >= atr_period else np.full_like(dm_plus, np.nan)
    dm_minus_sum = np.nancumsum(dm_minus) - np.concatenate([[0], np.nancumsum(dm_minus)[:-atr_period]]) if len(dm_minus) >= atr_period else np.full_like(dm_minus, np.nan)
    
    # Avoid division by zero
    tr_sum_safe = np.where(tr_sum == 0, 1e-10, tr_sum)
    
    # Calculate DI+ and DI-
    di_plus = 100 * dm_plus_sum / tr_sum_safe
    di_minus = 100 * dm_minus_sum / tr_sum_safe
    
    # Calculate DX
    dx = 100 * np.abs(di_plus - di_minus) / np.where((di_plus + di_minus) == 0, 1e-10, (di_plus + di_minus))
    
    # Calculate ADX (smoothed DX)
    adx = np.concatenate([[np.nan] * atr_period, 
                          np.nancumsum(dx[atr_period:]) / np.arange(1, len(dx[atr_period:]) + 1)])
    # Ensure we have at least atr_period values
    if len(adx) < 2 * atr_period:
        adx = np.full_like(dx, np.nan)
    else:
        # Apply additional smoothing for ADX
        adx_smoothed = np.concatenate([[np.nan] * (2 * atr_period - 1),
                                       np.nancumsum(adx[2 * atr_period - 1:]) / np.arange(1, len(adx[2 * atr_period - 1:]) + 1)])
        adx = adx_smoothed if len(adx_smoothed) == len(dx) else adx
    
    # Align ADX to 6h timeframe
    adx_aligned = align_htf_to_ltf(prices, df_1d, adx)
    
    # Calculate RSI on daily timeframe for momentum confirmation
    delta = np.diff(close_1d)
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    
    # First average gain/loss
    avg_gain = np.concatenate([[np.nan] * 14, 
                               [np.mean(gain[:14])] + [np.nan] * (len(gain) - 15)])
    avg_loss = np.concatenate([[np.nan] * 14, 
                               [np.mean(loss[:14])] + [np.nan] * (len(loss) - 15)])
    
    # Wilder smoothing for subsequent values
    for i in range(15, len(gain)):
        if not np.isnan(avg_gain[i-1]) and not np.isnan(gain[i]):
            avg_gain[i] = (avg_gain[i-1] * 13 + gain[i]) / 14
        else:
            avg_gain[i] = np.nan
        if not np.isnan(avg_loss[i-1]) and not np.isnan(loss[i]):
            avg_loss[i] = (avg_loss[i-1] * 13 + loss[i]) / 14
        else:
            avg_loss[i] = np.nan
    
    # Calculate RS and RSI
    rs = np.where(avg_loss != 0, avg_gain / avg_loss, 0)
    rsi = 100 - (100 / (1 + rs))
    rsi = np.concatenate([[np.nan] * 14, rsi[14:]])
    
    # Align RSI to 6h timeframe
    rsi_aligned = align_htf_to_ltf(prices, df_1d, rsi)
    
    # Volume confirmation on 6h timeframe
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_ratio = volume / np.where(vol_ma > 0, vol_ma, 1.0)
    vol_ratio = np.nan_to_num(vol_ratio, nan=1.0)
    
    signals = np.zeros(n)
    position = 0
    
    start_idx = 100  # Ensure sufficient data for indicators
    
    for i in range(start_idx, n):
        if (np.isnan(adx_aligned[i]) or np.isnan(rsi_aligned[i]) or 
            np.isnan(vol_ratio[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: Strong trend (ADX > 25), bullish momentum (RSI > 50), and volume confirmation
            if (adx_aligned[i] > 25 and 
                rsi_aligned[i] > 50 and
                vol_ratio[i] > 1.3):
                signals[i] = 0.25
                position = 1
            # Short: Strong trend (ADX > 25), bearish momentum (RSI < 50), and volume confirmation
            elif (adx_aligned[i] > 25 and 
                  rsi_aligned[i] < 50 and
                  vol_ratio[i] > 1.3):
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: Weakening trend (ADX < 20) or bearish momentum (RSI < 40)
            if adx_aligned[i] < 20 or rsi_aligned[i] < 40:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: Weakening trend (ADX < 20) or bullish momentum (RSI > 60)
            if adx_aligned[i] < 20 or rsi_aligned[i] > 60:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals
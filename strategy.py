#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h strategy using 1d RSI mean reversion with volume spike and ADX trend filter.
# Long when RSI(14) < 30 AND 4h volume > 1.5x 20-period average AND daily ADX > 25.
# Short when RSI(14) > 70 AND 4h volume > 1.5x 20-period average AND daily ADX > 25.
# Exit when RSI crosses back above 50 (long) or below 50 (short).
# RSI captures overextended moves, volume confirms participation, ADX ensures trending environment.
# Designed to work in both bull (buy dips) and bear (sell rallies) markets with controlled trade frequency.

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Calculate 4h RSI (14-period)
    delta = np.diff(close, prepend=close[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    
    # Wilder's smoothing for RSI
    def wilders_rsi(gain, loss, period):
        avg_gain = np.full_like(gain, np.nan)
        avg_loss = np.full_like(loss, np.nan)
        if len(gain) < period + 1:
            return avg_gain, avg_loss
        # First average: simple mean
        avg_gain[period] = np.mean(gain[1:period+1])
        avg_loss[period] = np.mean(loss[1:period+1])
        # Subsequent: Wilder's smoothing
        for i in range(period+1, len(gain)):
            avg_gain[i] = (avg_gain[i-1] * (period-1) + gain[i]) / period
            avg_loss[i] = (avg_loss[i-1] * (period-1) + loss[i]) / period
        rs = avg_gain / avg_loss
        rsi = 100 - (100 / (1 + rs))
        return rsi
    
    rsi = wilders_rsi(gain, loss, 14)
    
    # Calculate 4h volume moving average (20-period)
    vol_ma_20 = np.full_like(volume, np.nan)
    for i in range(19, len(volume)):
        vol_ma_20[i] = np.mean(volume[i-19:i+1])
    
    # Load daily data ONCE for ADX
    df_daily = get_htf_data(prices, '1d')
    if len(df_daily) < 30:
        return np.zeros(n)
    
    high_daily = df_daily['high'].values
    low_daily = df_daily['low'].values
    close_daily = df_daily['close'].values
    
    # Calculate daily ADX (14-period)
    # True Range
    tr1 = high_daily[1:] - low_daily[1:]
    tr2 = np.abs(high_daily[1:] - close_daily[:-1])
    tr3 = np.abs(low_daily[1:] - close_daily[:-1])
    tr = np.concatenate([[np.nan], np.maximum(tr1, np.maximum(tr2, tr3))])
    
    # Directional Movement
    dm_plus = np.where((high_daily[1:] - high_daily[:-1]) > (low_daily[:-1] - low_daily[1:]), 
                       np.maximum(high_daily[1:] - high_daily[:-1], 0), 0)
    dm_minus = np.where((low_daily[:-1] - low_daily[1:]) > (high_daily[1:] - high_daily[:-1]), 
                        np.maximum(low_daily[:-1] - low_daily[1:], 0), 0)
    dm_plus = np.concatenate([[0], dm_plus])
    dm_minus = np.concatenate([[0], dm_minus])
    
    # Wilder's smoothing
    def wilders_smoothing(data, period):
        result = np.full_like(data, np.nan)
        if len(data) < period:
            return result
        # First value: simple average (skip first element for DM)
        result[period-1] = np.nanmean(data[1:period])  # Skip first element which is 0 for DM
        for i in range(period, len(data)):
            result[i] = (result[i-1] * (period-1) + data[i]) / period
        return result
    
    tr_14 = wilders_smoothing(tr, 14)
    dm_plus_14 = wilders_smoothing(dm_plus, 14)
    dm_minus_14 = wilders_smoothing(dm_minus, 14)
    
    # DI+ and DI-
    di_plus = 100 * dm_plus_14 / tr_14
    di_minus = 100 * dm_minus_14 / tr_14
    
    # DX and ADX
    dx = 100 * np.abs(di_plus - di_minus) / (di_plus + di_minus)
    adx = np.full_like(dx, np.nan)
    # First ADX: simple average of first 14 DX values
    valid_dx = dx[~np.isnan(dx)]
    if len(valid_dx) >= 14:
        adx[13] = np.mean(valid_dx[:14])
        for i in range(14, len(dx)):
            if not np.isnan(dx[i]):
                adx[i] = (adx[i-1] * 13 + dx[i]) / 14
    
    # Align indicators to 4h timeframe
    rsi_aligned = align_htf_to_ltf(prices, pd.DataFrame({'close': close}), rsi)
    vol_ma_20_aligned = align_htf_to_ltf(prices, pd.DataFrame({'volume': volume}), vol_ma_20)
    adx_aligned = align_htf_to_ltf(prices, df_daily, adx)
    
    signals = np.zeros(n)
    position = 0
    position_size = 0.25  # 25% position size
    
    # Start after enough data for calculations
    start = max(30, 20)  # Need RSI and volume data
    
    for i in range(start, n):
        # Skip if any critical data is NaN
        if (np.isnan(rsi_aligned[i]) or 
            np.isnan(vol_ma_20_aligned[i]) or
            np.isnan(adx_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Volume ratio: current 4h volume vs 20-period average
        volume_ratio = volume[i] / vol_ma_20_aligned[i] if vol_ma_20_aligned[i] > 0 else 0
        
        if position == 0:
            # Look for mean reversion entries with volume confirmation and trend
            # Long: RSI < 30 (oversold) AND volume > 1.5x average AND ADX > 25
            if (rsi_aligned[i] < 30 and 
                volume_ratio > 1.5 and 
                adx_aligned[i] > 25):
                position = 1
                signals[i] = position_size
            # Short: RSI > 70 (overbought) AND volume > 1.5x average AND ADX > 25
            elif (rsi_aligned[i] > 70 and 
                  volume_ratio > 1.5 and 
                  adx_aligned[i] > 25):
                position = -1
                signals[i] = -position_size
            else:
                signals[i] = 0.0
        elif position == 1:
            # Exit long: RSI crosses back above 50 (mean reversion complete)
            if rsi_aligned[i] > 50:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = position_size
        elif position == -1:
            # Exit short: RSI crosses back below 50 (mean reversion complete)
            if rsi_aligned[i] < 50:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -position_size
    
    return signals

name = "4h_1d_RSI_MeanReversion_Volume_ADX_v1"
timeframe = "4h"
leverage = 1.0
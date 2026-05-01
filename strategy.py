#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h Camarilla R3/S3 breakout with 1d volume spike and choppiness regime filter.
# Camarilla pivot levels provide institutional support/resistance. Breakout above R3 or below S3
# with volume > 2x 20-period average and choppy market (CHOP > 61.8) indicates strong momentum
# in ranging conditions. Uses discrete sizing 0.30 to balance return and fee drag.
# Works in bull (buy R3 breakout) and bear (sell S3 breakout) via volume and regime confirmation.

name = "12h_Camarilla_R3S3_1dVolumeSpike_ChopFilter_v1"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Pre-compute session hours for efficiency
    hours = pd.DatetimeIndex(prices["open_time"]).hour
    
    # Load 1d data ONCE before loop for Camarilla pivots and volume average
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    # Calculate 1d Camarilla pivot levels (R3, S3)
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    pivot = (high_1d + low_1d + close_1d) / 3
    range_1d = high_1d - low_1d
    r3 = pivot + (range_1d * 1.1 / 2)
    s3 = pivot - (range_1d * 1.1 / 2)
    
    # Align Camarilla levels to 12h timeframe (wait for 1d close)
    r3_aligned = align_htf_to_ltf(prices, df_1d, r3)
    s3_aligned = align_htf_to_ltf(prices, df_1d, s3)
    
    # Calculate 1d volume moving average (20-period)
    vol_ma_1d = pd.Series(df_1d['volume'].values).rolling(window=20, min_periods=20).mean().values
    vol_ma_1d_aligned = align_htf_to_ltf(prices, df_1d, vol_ma_1d)
    
    # Calculate 12h Choppiness Index (CHOP) - regime filter
    def calculate_chop(high_arr, low_arr, close_arr, period=14):
        # True Range
        tr1 = np.abs(high_arr[1:] - low_arr[1:])
        tr2 = np.abs(high_arr[1:] - close_arr[:-1])
        tr3 = np.abs(low_arr[1:] - close_arr[:-1])
        tr = np.maximum(tr1, np.maximum(tr2, tr3))
        tr = np.concatenate([[np.nan], tr])  # align length
        
        # ATR = smoothed TR (using Wilder's smoothing)
        def wilders_smoothing(data, period):
            result = np.full_like(data, np.nan)
            if len(data) < period:
                return result
            # First value is simple average
            result[period-1] = np.nanmean(data[:period])
            # Subsequent values: Wilder's smoothing
            for i in range(period, len(data)):
                result[i] = (result[i-1] * (period-1) + data[i]) / period
            return result
        
        atr = wilders_smoothing(tr, period)
        
        # Sum of TR over period
        sum_tr = np.nansum(tr.reshape(-1, period), axis=1) if len(tr) >= period else np.full_like(tr, np.nan)
        # For simplicity, use rolling sum with min_periods
        sum_tr_rolling = pd.Series(tr).rolling(window=period, min_periods=period).sum().values
        
        # CHOP = 100 * log10(sum_tr_rolling / (atr * period)) / log10(period)
        # Avoid division by zero and handle NaN
        denominator = atr * period
        chop = np.full_like(tr, np.nan)
        mask = (denominator > 0) & ~np.isnan(denominator) & ~np.isnan(sum_tr_rolling)
        chop[mask] = 100 * np.log10(sum_tr_rolling[mask] / denominator[mask]) / np.log10(period)
        return chop
    
    chop_values = calculate_chop(high, low, close, 14)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 50  # warmup for indicators
    
    for i in range(start_idx, n):
        # Session filter: trade all sessions for 12h timeframe
        hour = hours[i]
        
        if np.isnan(r3_aligned[i]) or np.isnan(s3_aligned[i]) or np.isnan(vol_ma_1d_aligned[i]) or np.isnan(chop_values[i]):
            signals[i] = 0.0
            continue
        
        curr_close = close[i]
        curr_high = high[i]
        curr_low = low[i]
        curr_vol = volume[i]
        curr_r3 = r3_aligned[i]
        curr_s3 = s3_aligned[i]
        curr_vol_ma = vol_ma_1d_aligned[i]
        curr_chop = chop_values[i]
        
        # Volume confirmation: current 12h volume > 2.0x 1d volume MA
        if curr_vol_ma <= 0:
            signals[i] = 0.0
            continue
        volume_confirm = curr_vol > (curr_vol_ma * 2.0)
        
        # Regime filter: choppy market (CHOP > 61.8) for mean reversion breakdown into trend
        regime_filter = curr_chop > 61.8
        
        # Breakout conditions
        long_breakout = curr_high > curr_r3
        short_breakout = curr_low < curr_s3
        
        # Entry conditions
        if position == 0:  # Flat - look for new entries
            # Long: price breaks above R3 AND volume confirmation AND choppy regime
            if (long_breakout and 
                volume_confirm and 
                regime_filter):
                signals[i] = 0.30
                position = 1
            # Short: price breaks below S3 AND volume confirmation AND choppy regime
            elif (short_breakout and 
                  volume_confirm and 
                  regime_filter):
                signals[i] = -0.30
                position = -1
            else:
                signals[i] = 0.0
        
        elif position == 1:  # Long position
            # Exit: price breaks below S3 (failed breakout) OR regime shifts to trending (CHOP < 38.2)
            if (curr_low < curr_s3 or 
                curr_chop < 38.2):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.30
        
        elif position == -1:  # Short position
            # Exit: price breaks above R3 (failed breakout) OR regime shifts to trending (CHOP < 38.2)
            if (curr_high > curr_r3 or 
                curr_chop < 38.2):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.30
    
    return signals
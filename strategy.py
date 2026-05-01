#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Bollinger Band Squeeze Breakout with 1d ADX trend filter and volume confirmation.
# Long when price breaks above upper BB(20,2) during low volatility (BB Width < 20th percentile) 
# AND 1d ADX > 25 (trending market) AND volume > 1.5x 20-bar average.
# Short when price breaks below lower BB(20,2) under same conditions.
# Uses discrete sizing 0.25 to minimize fee churn. Designed for 6h timeframe to capture 
# volatility expansion moves in both bull and bear markets via ADX filter.

name = "6h_BB_Squeeze_Breakout_1dADX_VolumeConfirm_v1"
timeframe = "6h"
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
    
    # Load 1d data ONCE before loop for ADX trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    # 1d ADX calculation (14-period)
    def calculate_adx(high_arr, low_arr, close_arr, period=14):
        # True Range
        tr1 = np.abs(high_arr[1:] - low_arr[1:])
        tr2 = np.abs(high_arr[1:] - close_arr[:-1])
        tr3 = np.abs(low_arr[1:] - close_arr[:-1])
        tr = np.maximum(tr1, np.maximum(tr2, tr3))
        tr = np.concatenate([[np.nan], tr])  # align length
        
        # Directional Movement
        up_move = high_arr[1:] - high_arr[:-1]
        down_move = low_arr[:-1] - low_arr[1:]
        plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0.0)
        minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0.0)
        plus_dm = np.concatenate([[0.0], plus_dm])
        minus_dm = np.concatenate([[0.0], minus_dm])
        
        # Smoothed TR, +DM, -DM using Wilder's smoothing (EMA with alpha=1/period)
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
        plus_dm_smooth = wilders_smoothing(plus_dm, period)
        minus_dm_smooth = wilders_smoothing(minus_dm, period)
        
        # Directional Indicators
        plus_di = 100 * plus_dm_smooth / atr
        minus_di = 100 * minus_dm_smooth / atr
        
        # DX and ADX
        dx = 100 * np.abs(plus_di - minus_di) / (plus_di + minus_di)
        adx = wilders_smoothing(dx, period)
        return adx
    
    # Calculate 1d ADX
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    adx_1d = calculate_adx(high_1d, low_1d, close_1d, 14)
    adx_1d_aligned = align_htf_to_ltf(prices, df_1d, adx_1d)
    
    # Bollinger Bands on 6h data (20,2)
    bb_period = 20
    bb_std = 2
    sma = pd.Series(close).rolling(window=bb_period, min_periods=bb_period).mean()
    std = pd.Series(close).rolling(window=bb_period, min_periods=bb_period).std()
    upper_band = sma + (bb_std * std)
    lower_band = sma - (bb_std * std)
    bb_width = upper_band - lower_band
    
    upper_band_values = upper_band.values
    lower_band_values = lower_band.values
    bb_width_values = bb_width.values
    sma_values = sma.values
    
    # BB Width percentile (20th percentile lookback for squeeze detection)
    bb_width_percentile = pd.Series(bb_width_values).rolling(window=50, min_periods=20).quantile(0.20)
    bb_width_percentile_values = bb_width_percentile.values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 60  # warmup for BB and ADX
    
    for i in range(start_idx, n):
        # Session filter: trade all sessions for 6h timeframe
        hour = hours[i]
        
        if (np.isnan(adx_1d_aligned[i]) or np.isnan(upper_band_values[i]) or 
            np.isnan(lower_band_values[i]) or np.isnan(bb_width_percentile_values[i])):
            signals[i] = 0.0
            continue
        
        curr_close = close[i]
        curr_high = high[i]
        curr_low = low[i]
        curr_vol = volume[i]
        curr_adx = adx_1d_aligned[i]
        curr_upper = upper_band_values[i]
        curr_lower = lower_band_values[i]
        curr_width = bb_width_values[i]
        curr_width_pctl = bb_width_percentile_values[i]
        
        # Volume confirmation: current 6h volume > 1.5x 20-period average
        if i < 20 + start_idx:
            signals[i] = 0.0
            continue
            
        vol_ma = np.mean(volume[i-20:i])  # 20-period simple moving average
        if vol_ma <= 0:
            signals[i] = 0.0
            continue
        volume_confirm = curr_vol > (vol_ma * 1.5)
        
        # Bollinger Squeeze condition: BB Width < 20th percentile (low volatility)
        squeeze_condition = curr_width < curr_width_pctl
        
        # Breakout conditions
        long_breakout = curr_high > curr_upper  # price breaks above upper BB
        short_breakout = curr_low < curr_lower  # price breaks below lower BB
        
        # Entry conditions
        if position == 0:  # Flat - look for new entries
            # Long: bullish breakout during squeeze AND ADX > 25 AND volume confirmation
            if (squeeze_condition and long_breakout and 
                curr_adx > 25 and volume_confirm):
                signals[i] = 0.25
                position = 1
            # Short: bearish breakout during squeeze AND ADX > 25 AND volume confirmation
            elif (squeeze_condition and short_breakout and 
                  curr_adx > 25 and volume_confirm):
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        
        elif position == 1:  # Long position
            # Exit: price returns to middle BB (mean reversion) OR ADX < 20 (trend weakening)
            if (curr_close <= sma_values[i] or curr_adx < 20):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:  # Short position
            # Exit: price returns to middle BB (mean reversion) OR ADX < 20 (trend weakening)
            if (curr_close >= sma_values[i] or curr_adx < 20):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals
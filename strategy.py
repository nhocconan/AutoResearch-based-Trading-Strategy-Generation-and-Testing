#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Bollinger Band Squeeze Breakout with 1d ADX(14) regime filter and volume confirmation
# Bollinger Band Width (BBW) < 20th percentile indicates low volatility squeeze
# Breakout occurs when price closes outside BB(20,2) with volume > 1.5x 20-period average
# 1d ADX(14) > 25 confirms trending regime (avoid ranging markets)
# Designed for low trade frequency (12-37/year) to minimize fee drag
# Works in both bull and bear markets by trading breakouts in the direction of the 1d trend

name = "6h_BB_Squeeze_Breakout_1dADX_Volume_v1"
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
    
    # Get 1d data for ADX(14) regime filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 14:
        return np.zeros(n)
    
    # Calculate ADX(14) on 1d for regime filter
    # ADX requires +DI, -DI, and DX calculations
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # True Range
    tr1 = np.abs(high_1d[1:] - low_1d[1:])
    tr2 = np.abs(high_1d[1:] - close_1d[:-1])
    tr3 = np.abs(low_1d[1:] - close_1d[:-1])
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr = np.concatenate([[np.nan], tr])  # prepend NaN for first element
    
    # +DM and -DM
    up_move = high_1d[1:] - high_1d[:-1]
    down_move = low_1d[:-1] - low_1d[1:]
    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0.0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0.0)
    plus_dm = np.concatenate([[0.0], plus_dm])
    minus_dm = np.concatenate([[0.0], minus_dm])
    
    # Smoothed TR, +DM, -DM (Wilder's smoothing = EMA with alpha=1/period)
    def wilders_smooth(data, period):
        result = np.full_like(data, np.nan)
        if len(data) < period:
            return result
        # First value is simple average
        result[period-1] = np.nansum(data[1:period]) / (period-1) if period > 1 else data[0]
        # Subsequent values: Wilder's smoothing
        for i in range(period, len(data)):
            result[i] = (result[i-1] * (period-1) + data[i]) / period
        return result
    
    tr_smooth = wilders_smooth(tr, 14)
    plus_dm_smooth = wilders_smooth(plus_dm, 14)
    minus_dm_smooth = wilders_smooth(minus_dm, 14)
    
    # +DI and -DI
    plus_di = np.where(tr_smooth != 0, (plus_dm_smooth / tr_smooth) * 100, 0)
    minus_di = np.where(tr_smooth != 0, (minus_dm_smooth / tr_smooth) * 100, 0)
    
    # DX and ADX
    dx = np.where((plus_di + minus_di) != 0, np.abs(plus_di - minus_di) / (plus_di + minus_di) * 100, 0)
    adx = wilders_smooth(dx, 14)
    
    # Align 1d ADX to 6h timeframe (wait for completed 1d bar)
    adx_1d_aligned = align_htf_to_ltf(prices, df_1d, adx)
    
    # Bollinger Bands (20,2) on 6h
    bb_period = 20
    bb_std = 2
    sma = pd.Series(close).rolling(window=bb_period, min_periods=bb_period).mean().values
    std_dev = pd.Series(close).rolling(window=bb_period, min_periods=bb_period).std().values
    upper_band = sma + (bb_std * std_dev)
    lower_band = sma - (bb_std * std_dev)
    
    # Bollinger Band Width
    bb_width = (upper_band - lower_band) / sma * 100  # as percentage
    
    # BBW percentile lookback (50 periods) for squeeze detection
    bb_width_series = pd.Series(bb_width)
    bb_width_percentile = bb_width_series.rolling(window=50, min_periods=20).quantile(0.2).values
    squeeze_condition = bb_width < bb_width_percentile
    
    # Volume confirmation (1.5x 20-period average) on 6h
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().shift(1).values
    volume_spike = volume > (vol_ma * 1.5)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after warmup (need enough for all calculations)
    start_idx = max(50, 34)  # BBW percentile lookback and ADX smoothing
    
    for i in range(start_idx, n):
        # Check for NaN values in indicators
        if (np.isnan(adx_1d_aligned[i]) or np.isnan(squeeze_condition[i]) or 
            np.isnan(volume_spike[i]) or np.isnan(upper_band[i]) or np.isnan(lower_band[i])):
            signals[i] = 0.0
            continue
        
        if position == 0:  # Flat - look for new entries
            # Long entry: BB squeeze breakout above upper band + ADX > 25 (trending) + volume spike
            if (close[i] > upper_band[i] and squeeze_condition[i] and 
                adx_1d_aligned[i] > 25 and volume_spike[i]):
                signals[i] = 0.25
                position = 1
            # Short entry: BB squeeze breakout below lower band + ADX > 25 (trending) + volume spike
            elif (close[i] < lower_band[i] and squeeze_condition[i] and 
                  adx_1d_aligned[i] > 25 and volume_spike[i]):
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        
        elif position == 1:  # Long position
            # Exit: price closes below middle band (SMA) or ADX drops below 20 (trend weakening)
            if close[i] < sma[i] or adx_1d_aligned[i] < 20:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:  # Short position
            # Exit: price closes above middle band (SMA) or ADX drops below 20 (trend weakening)
            if close[i] > sma[i] or adx_1d_aligned[i] < 20:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals
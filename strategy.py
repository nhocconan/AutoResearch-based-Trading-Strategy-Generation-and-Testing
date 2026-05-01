#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Camarilla R3/S3 Breakout with 1d Volume Spike and ADX Trend Filter
# Uses 1d Camarilla levels (R3/S3) for breakout entries, confirmed by 1d volume spike (>1.5x 20-period average) and 1d ADX>25 for trend strength
# In trending regime (ADX>25): breakout continuation trades
# In ranging regime (ADX<20): fade at R3/S3 levels
# Designed for low frequency (50-150 trades over 4 years) with clear entry/exit rules

name = "6h_Camarilla_R3S3_1dADX_Volume_Regime_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # 1d HTF data for regime and level calculation
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    volume_1d = df_1d['volume'].values
    
    # 1d ADX(14) calculation for regime detection
    def calculate_adx(high, low, close, period=14):
        # True Range
        tr1 = np.abs(high[1:] - low[1:])
        tr2 = np.abs(high[1:] - close[:-1])
        tr3 = np.abs(low[1:] - close[:-1])
        tr = np.maximum(tr1, np.maximum(tr2, tr3))
        tr = np.concatenate([[np.nan], tr])
        
        # Directional Movement
        dm_plus = np.where((high[1:] - high[:-1]) > (low[:-1] - low[1:]), 
                           np.maximum(high[1:] - high[:-1], 0), 0)
        dm_minus = np.where((low[:-1] - low[1:]) > (high[1:] - high[:-1]), 
                            np.maximum(low[:-1] - low[1:], 0), 0)
        dm_plus = np.concatenate([[0], dm_plus])
        dm_minus = np.concatenate([[0], dm_minus])
        
        # Wilder's smoothing
        def wilders_smoothing(x, period):
            result = np.full_like(x, np.nan)
            if len(x) >= period:
                first_val = np.nansum(x[1:period+1])
                result[period] = first_val
                for i in range(period+1, len(x)):
                    result[i] = result[i-1] - (result[i-1] / period) + x[i]
            return result
        
        tr_smoothed = wilders_smoothing(tr, period)
        dm_plus_smoothed = wilders_smoothing(dm_plus, period)
        dm_minus_smoothed = wilders_smoothing(dm_minus, period)
        
        # DI+ and DI-
        di_plus = np.where(tr_smoothed != 0, (dm_plus_smoothed / tr_smoothed) * 100, 0)
        di_minus = np.where(tr_smoothed != 0, (dm_minus_smoothed / tr_smoothed) * 100, 0)
        
        # DX and ADX
        dx = np.where((di_plus + di_minus) != 0, 
                      np.abs(di_plus - di_minus) / (di_plus + di_minus) * 100, 0)
        adx = wilders_smoothing(dx, period)
        return adx
    
    adx_1d = calculate_adx(high_1d, low_1d, close_1d, 14)
    adx_aligned = align_htf_to_ltf(prices, df_1d, adx_1d)
    
    # 1d Volume Spike (>1.5x 20-period average)
    def calculate_volume_spike(volume, period=20, threshold=1.5):
        vol_ma = pd.Series(volume).rolling(window=period, min_periods=period).mean().values
        volume_ratio = np.where(vol_ma > 0, volume / vol_ma, 0)
        return volume_ratio > threshold
    
    volume_spike_1d = calculate_volume_spike(volume_1d, 20, 1.5)
    volume_spike_aligned = align_htf_to_ltf(prices, df_1d, volume_spike_1d.astype(float))
    
    # 1d Camarilla Levels (based on previous day's OHLC)
    def calculate_camarilla(high, low, close):
        # Camarilla levels: R4, R3, R2, R1, PP, S1, S2, S3, S4
        # R4 = close + ((high-low) * 1.1/2)
        # R3 = close + ((high-low) * 1.1/4)
        # S3 = close - ((high-low) * 1.1/4)
        # S4 = close - ((high-low) * 1.1/2)
        rango = high - low
        r3 = close + (rango * 1.1 / 4)
        s3 = close - (rango * 1.1 / 4)
        return r3, s3
    
    # Calculate Camarilla for each 1d bar (using previous day's data)
    r3_1d = np.full_like(close_1d, np.nan)
    s3_1d = np.full_like(close_1d, np.nan)
    
    for i in range(1, len(close_1d)):
        r3_1d[i], s3_1d[i] = calculate_camarilla(high_1d[i-1], low_1d[i-1], close_1d[i-1])
    
    r3_aligned = align_htf_to_ltf(prices, df_1d, r3_1d)
    s3_aligned = align_htf_to_ltf(prices, df_1d, s3_1d)
    
    # 6h EMA20 for dynamic filtering
    ema20 = pd.Series(close).ewm(span=20, adjust=False, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after warmup
    start_idx = max(34, 20)  # Need ADX and EMA20
    
    for i in range(start_idx, n):
        if (np.isnan(adx_aligned[i]) or np.isnan(volume_spike_aligned[i]) or 
            np.isnan(r3_aligned[i]) or np.isnan(s3_aligned[i]) or np.isnan(ema20[i])):
            signals[i] = 0.0
            continue
        
        # Regime filters
        trending = adx_aligned[i] > 25
        ranging = adx_aligned[i] < 20
        
        if position == 0:  # Flat - look for new entries
            # Volume confirmation required
            if volume_spike_aligned[i]:
                if trending:
                    # Trending regime: breakout continuation
                    # Long: price breaks above R3 and above EMA20
                    if close[i] > r3_aligned[i] and close[i] > ema20[i]:
                        signals[i] = 0.25
                        position = 1
                    # Short: price breaks below S3 and below EMA20
                    elif close[i] < s3_aligned[i] and close[i] < ema20[i]:
                        signals[i] = -0.25
                        position = -1
                elif ranging:
                    # Ranging regime: fade at extremes
                    # Long: price touches S3 and shows rejection (close > open)
                    if close[i] <= s3_aligned[i] and close[i] > prices['open'].iloc[i]:
                        signals[i] = 0.25
                        position = 1
                    # Short: price touches R3 and shows rejection (close < open)
                    elif close[i] >= r3_aligned[i] and close[i] < prices['open'].iloc[i]:
                        signals[i] = -0.25
                        position = -1
            else:
                signals[i] = 0.0  # No volume spike - stay flat
        
        elif position == 1:  # Long position
            # Exit conditions
            exit_long = False
            if trending:
                # Exit trending long when price falls below S3 or EMA20
                if close[i] < s3_aligned[i] or close[i] < ema20[i]:
                    exit_long = True
            elif ranging:
                # Exit ranging long when price reaches R3 (mean reversion target)
                if close[i] >= r3_aligned[i]:
                    exit_long = True
            else:
                # Transition regime - exit on any adverse move
                if close[i] < ema20[i]:
                    exit_long = True
            
            if exit_long:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:  # Short position
            # Exit conditions
            exit_short = False
            if trending:
                # Exit trending short when price rises above R3 or EMA20
                if close[i] > r3_aligned[i] or close[i] > ema20[i]:
                    exit_short = True
            elif ranging:
                # Exit ranging short when price reaches S3 (mean reversion target)
                if close[i] <= s3_aligned[i]:
                    exit_short = True
            else:
                # Transition regime - exit on any adverse move
                if close[i] > ema20[i]:
                    exit_short = True
            
            if exit_short:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals
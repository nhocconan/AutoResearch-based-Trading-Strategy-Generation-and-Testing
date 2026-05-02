#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Elder Ray (Bull/Bear Power) with 1d regime filter and volume confirmation
# Elder Ray: Bull Power = High - EMA13, Bear Power = Low - EMA13 (using 13-period EMA)
# 1d ADX > 25 determines trending regime (follow Elder Ray signals)
# 1d ADX < 20 determines ranging regime (fade Elder Ray extremes)
# Volume spike (1.5x 20-bar MA) confirms institutional participation
# Designed for 50-150 total trades over 4 years (12-37/year) on 6h timeframe
# Works in bull markets (trend following) and bear markets (mean reversion in ranges)

name = "6h_ElderRay_1dADX_Regime_Volume"
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
    
    # Calculate 1d ADX for regime filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:  # Need enough data for ADX calculation
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate ADX components
    plus_dm = np.where((high_1d[1:] - high_1d[:-1]) > (low_1d[:-1] - low_1d[1:]), 
                       np.maximum(high_1d[1:] - high_1d[:-1], 0), 0)
    minus_dm = np.where((low_1d[:-1] - low_1d[1:]) > (high_1d[1:] - high_1d[:-1]), 
                        np.maximum(low_1d[:-1] - low_1d[1:], 0), 0)
    # Pad to original length
    plus_dm = np.concatenate([[0], plus_dm])
    minus_dm = np.concatenate([[0], minus_dm])
    
    tr1 = high_1d - low_1d
    tr2 = np.abs(high_1d - np.concatenate([[close_1d[0]], close_1d[:-1]]))
    tr3 = np.abs(low_1d - np.concatenate([[close_1d[0]], close_1d[:-1]]))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    
    # Wilder's smoothing (alpha = 1/period)
    def wilders_smoothing(data, period):
        result = np.zeros_like(data)
        result[period-1] = np.mean(data[:period])
        for i in range(period, len(data)):
            result[i] = (result[i-1] * (period-1) + data[i]) / period
        return result
    
    period = 14
    atr = wilders_smoothing(tr, period)
    plus_di = 100 * wilders_smoothing(plus_dm, period) / (atr + 1e-10)
    minus_di = 100 * wilders_smoothing(minus_dm, period) / (atr + 1e-10)
    dx = 100 * np.abs(plus_di - minus_di) / (plus_di + minus_di + 1e-10)
    adx = wilders_smoothing(dx, period)
    
    # Align ADX to 6h timeframe
    adx_aligned = align_htf_to_ltf(prices, df_1d, adx)
    
    # Calculate 13-period EMA for Elder Ray (on 6h data)
    close_s = pd.Series(close)
    ema13 = close_s.ewm(span=13, adjust=False, min_periods=13).mean().values
    
    # Elder Ray components
    bull_power = high - ema13
    bear_power = low - ema13
    
    # Volume confirmation: 1.5x 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (1.5 * vol_ma)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after warmup
    start_idx = 50
    
    for i in range(start_idx, n):
        # Check for NaN values in indicators
        if (np.isnan(adx_aligned[i]) or np.isnan(ema13[i]) or 
            np.isnan(bull_power[i]) or np.isnan(bear_power[i]) or 
            np.isnan(vol_ma[i])):
            signals[i] = 0.0
            continue
        
        if position == 0:  # Flat - look for new entries
            # Determine regime based on 1d ADX
            if adx_aligned[i] > 25:  # Trending regime - follow Elder Ray
                # Long: Bull Power > 0 AND volume spike
                if bull_power[i] > 0 and volume_spike[i]:
                    signals[i] = 0.25
                    position = 1
                # Short: Bear Power < 0 AND volume spike
                elif bear_power[i] < 0 and volume_spike[i]:
                    signals[i] = -0.25
                    position = -1
                else:
                    signals[i] = 0.0
            elif adx_aligned[i] < 20:  # Ranging regime - fade Elder Ray extremes
                # Long: Bear Power < -threshold (oversold) AND volume spike
                # Short: Bull Power > threshold (overbought) AND volume spike
                # Use dynamic threshold based on recent volatility
                lookback = min(50, i)
                if lookback >= 10:
                    bp_std = np.std(bear_power[i-lookback:i])
                    bull_std = np.std(bull_power[i-lookback:i])
                    threshold = 2.0  # 2 standard deviations
                    
                    if bear_power[i] < (-threshold * bp_std) and volume_spike[i]:
                        signals[i] = 0.25
                        position = 1
                    elif bull_power[i] > (threshold * bull_std) and volume_spike[i]:
                        signals[i] = -0.25
                        position = -1
                    else:
                        signals[i] = 0.0
                else:
                    signals[i] = 0.0
            else:  # Transition regime (ADX 20-25) - no trades
                signals[i] = 0.0
        
        elif position == 1:  # Long position
            # Exit conditions
            exit_signal = False
            if adx_aligned[i] > 25:  # Trending regime
                # Exit when Bear Power becomes negative (trend weakness)
                if bear_power[i] < 0:
                    exit_signal = True
            elif adx_aligned[i] < 20:  # Ranging regime
                # Exit when price returns to mean (EMA13)
                if close[i] >= ema13[i]:
                    exit_signal = True
            else:  # Transition regime
                # Exit on any contrary Elder Ray signal
                if bear_power[i] < 0:
                    exit_signal = True
            
            if exit_signal:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:  # Short position
            # Exit conditions
            exit_signal = False
            if adx_aligned[i] > 25:  # Trending regime
                # Exit when Bull Power becomes positive (trend weakness)
                if bull_power[i] > 0:
                    exit_signal = True
            elif adx_aligned[i] < 20:  # Ranging regime
                # Exit when price returns to mean (EMA13)
                if close[i] <= ema13[i]:
                    exit_signal = True
            else:  # Transition regime
                # Exit on any contrary Elder Ray signal
                if bull_power[i] > 0:
                    exit_signal = True
            
            if exit_signal:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals
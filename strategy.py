#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Williams Fractal Breakout + 1d Volume Spike + 1d ADX Regime Filter
# Williams Fractals identify swing highs/lows that require 2-bar confirmation (no look-ahead)
# Breakout above latest bullish fractal with volume spike = long
# Breakdown below latest bearish fractal with volume spike = short
# 1d ADX(20) filters regime: ADX>25 = trending (trade breakouts), ADX<20 = range (avoid)
# Volume spike: current volume > 1.5 * 20-period EMA of volume
# Designed for low frequency (50-150 trades over 4 years) with clear structure
# Works in bull/bear: captures momentum in trending markets, avoids chop

name = "6h_WilliamsFractal_1dVolume_1dADX_Regime_v1"
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
    
    # 1d HTF data for fractals, volume average, and ADX
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    
    # Calculate Williams Fractals (requires 2-bar confirmation)
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    
    # Bullish fractal: low[n] < low[n-1] and low[n] < low[n+1] and low[n+1] < low[n+2] and low[n-1] < low[n-2]
    # Bearish fractal: high[n] > high[n-1] and high[n] > high[n+1] and high[n+1] > high[n+2] and high[n-1] > high[n-2]
    # We need 2 extra bars for confirmation, so we'll use additional_delay_bars=2 in align_htf_to_ltf
    bullish_fractal = np.full(len(high_1d), np.nan)
    bearish_fractal = np.full(len(high_1d), np.nan)
    
    for i in range(2, len(high_1d) - 2):
        # Bullish fractal at i
        if (low_1d[i] < low_1d[i-1] and low_1d[i] < low_1d[i+1] and 
            low_1d[i+1] < low_1d[i+2] and low_1d[i-1] < low_1d[i-2]):
            bullish_fractal[i] = low_1d[i]
        # Bearish fractal at i
        if (high_1d[i] > high_1d[i-1] and high_1d[i] > high_1d[i+1] and 
            high_1d[i+1] > high_1d[i+2] and high_1d[i-1] > high_1d[i-2]):
            bearish_fractal[i] = high_1d[i]
    
    # Align fractals to 6h timeframe with 2-bar extra delay for confirmation
    bullish_fractal_aligned = align_htf_to_ltf(prices, df_1d, bullish_fractal, additional_delay_bars=2)
    bearish_fractal_aligned = align_htf_to_ltf(prices, df_1d, bearish_fractal, additional_delay_bars=2)
    
    # 1d volume spike filter: volume > 1.5 * 20-period EMA
    vol_ema_20 = pd.Series(df_1d['volume'].values).ewm(span=20, adjust=False, min_periods=20).mean().values
    volume_spike_1d = df_1d['volume'].values > (1.5 * vol_ema_20)
    volume_spike_aligned = align_htf_to_ltf(prices, df_1d, volume_spike_1d)
    
    # 1d ADX(20) for regime filter
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # True Range
    tr1 = np.abs(high_1d[1:] - low_1d[1:])
    tr2 = np.abs(high_1d[1:] - close_1d[:-1])
    tr3 = np.abs(low_1d[1:] - close_1d[:-1])
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr = np.concatenate([[np.nan], tr])
    
    # Directional Movement
    dm_plus = np.where((high_1d[1:] - high_1d[:-1]) > (low_1d[:-1] - low_1d[1:]), 
                       np.maximum(high_1d[1:] - high_1d[:-1], 0), 0)
    dm_minus = np.where((low_1d[:-1] - low_1d[1:]) > (high_1d[1:] - high_1d[:-1]), 
                        np.maximum(low_1d[:-1] - low_1d[1:], 0), 0)
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
    
    tr_period = 20
    tr_smoothed = wilders_smoothing(tr, tr_period)
    dm_plus_smoothed = wilders_smoothing(dm_plus, tr_period)
    dm_minus_smoothed = wilders_smoothing(dm_minus, tr_period)
    
    # DI+ and DI-
    di_plus = np.where(tr_smoothed != 0, (dm_plus_smoothed / tr_smoothed) * 100, 0)
    di_minus = np.where(tr_smoothed != 0, (dm_minus_smoothed / tr_smoothed) * 100, 0)
    
    # DX and ADX
    dx = np.where((di_plus + di_minus) != 0, 
                  np.abs(di_plus - di_minus) / (di_plus + di_minus) * 100, 0)
    adx = wilders_smoothing(dx, tr_period)
    adx_aligned = align_htf_to_ltf(prices, df_1d, adx)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after warmup
    start_idx = max(34, 20)  # Need ADX and EMA20
    
    for i in range(start_idx, n):
        if (np.isnan(bullish_fractal_aligned[i]) or np.isnan(bearish_fractal_aligned[i]) or 
            np.isnan(adx_aligned[i]) or np.isnan(volume_spike_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Regime filters
        trending = adx_aligned[i] > 25
        ranging = adx_aligned[i] < 20
        
        if position == 0:  # Flat - look for new entries
            # Only trade in trending regime (ADX>25) - avoid ranging markets
            if trending:
                # Long: Break above latest bullish fractal with volume spike
                if close[i] > bullish_fractal_aligned[i] and volume_spike_aligned[i]:
                    signals[i] = 0.25
                    position = 1
                # Short: Break below latest bearish fractal with volume spike
                elif close[i] < bearish_fractal_aligned[i] and volume_spike_aligned[i]:
                    signals[i] = -0.25
                    position = -1
                else:
                    signals[i] = 0.0
            else:
                signals[i] = 0.0  # Avoid ranging and transition regimes
        
        elif position == 1:  # Long position
            # Exit conditions: price returns to bearish fractal or opposite breakout
            exit_long = False
            if close[i] <= bearish_fractal_aligned[i]:  # Return to bearish fractal (support)
                exit_long = True
            elif close[i] < bearish_fractal_aligned[i] and volume_spike_aligned[i]:  # Reverse breakdown
                exit_long = True
            
            if exit_long:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:  # Short position
            # Exit conditions: price returns to bullish fractal or opposite breakout
            exit_short = False
            if close[i] >= bullish_fractal_aligned[i]:  # Return to bullish fractal (resistance)
                exit_short = True
            elif close[i] > bullish_fractal_aligned[i] and volume_spike_aligned[i]:  # Reverse breakout
                exit_short = True
            
            if exit_short:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals
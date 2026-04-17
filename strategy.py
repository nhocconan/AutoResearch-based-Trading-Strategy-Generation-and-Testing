#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h ADX(14) trend strength with 1d Parabolic SAR trend filter and volume confirmation.
# Uses ADX to filter for strong trends (ADX > 25) and Parabolic SAR for entry/exit signals.
# Volume spike confirms breakout strength. Designed to capture strong trends with low turnover.
# Target: 12-30 trades/year to stay within optimal range for 12h timeframe.
# Works in both bull and bear markets by following the trend direction.

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1d data for Parabolic SAR and ADX calculation
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate 1d Parabolic SAR (0.02 step, 0.2 max)
    psar = np.zeros(len(close_1d))
    psar[0] = low_1d[0]
    psarfactor = 0.02
    psar_max = 0.2
    psar_dir = 1  # 1 for uptrend, -1 for downtrend
    psar_ep = high_1d[0] if psar_dir == 1 else low_1d[0]
    
    for i in range(1, len(close_1d)):
        if psar_dir == 1:  # uptrend
            psar[i] = psar[i-1] + psarfactor * (psar_ep - psar[i-1])
            if psar[i] > low_1d[i]:  # trend reversal
                psar_dir = -1
                psar[i] = psar_ep
                psar_ep = low_1d[i]
                psarfactor = 0.02
            else:
                if high_1d[i] > psar_ep:
                    psar_ep = high_1d[i]
                    psarfactor = min(psarfactor + 0.02, psar_max)
        else:  # downtrend
            psar[i] = psar[i-1] - psarfactor * (psar[i-1] - psar_ep)
            if psar[i] < high_1d[i]:  # trend reversal
                psar_dir = 1
                psar[i] = psar_ep
                psar_ep = high_1d[i]
                psarfactor = 0.02
            else:
                if low_1d[i] < psar_ep:
                    psar_ep = low_1d[i]
                    psarfactor = min(psarfactor + 0.02, psar_max)
    
    # Calculate 1d ADX (14-period)
    # True Range
    tr1 = high_1d[1:] - low_1d[1:]
    tr2 = np.abs(high_1d[1:] - close_1d[:-1])
    tr3 = np.abs(low_1d[1:] - close_1d[:-1])
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr = np.insert(tr, 0, high_1d[0] - low_1d[0])  # first TR
    
    # Directional Movement
    dm_plus = np.where((high_1d[1:] - high_1d[:-1]) > (low_1d[:-1] - low_1d[1:]), 
                       np.maximum(high_1d[1:] - high_1d[:-1], 0), 0)
    dm_minus = np.where((low_1d[:-1] - low_1d[1:]) > (high_1d[1:] - high_1d[:-1]), 
                        np.maximum(low_1d[:-1] - low_1d[1:], 0), 0)
    dm_plus = np.insert(dm_plus, 0, 0)
    dm_minus = np.insert(dm_minus, 0, 0)
    
    # Smoothed values
    def WilderSmoothing(data, period):
        result = np.zeros_like(data)
        result[period-1] = np.nansum(data[:period])
        for i in range(period, len(data)):
            result[i] = result[i-1] - (result[i-1] / period) + data[i]
        return result
    
    atr = WilderSmoothing(tr, 14)
    dm_plus_smooth = WilderSmoothing(dm_plus, 14)
    dm_minus_smooth = WilderSmoothing(dm_minus, 14)
    
    # DI+ and DI-
    di_plus = 100 * dm_plus_smooth / atr
    di_minus = 100 * dm_minus_smooth / atr
    
    # DX and ADX
    dx = 100 * np.abs(di_plus - di_minus) / (di_plus + di_minus)
    dx = np.where((di_plus + di_minus) == 0, 0, dx)
    adx = WilderSmoothing(dx, 14)
    
    # Align 1d indicators to 12h
    psar_12h = align_htf_to_ltf(prices, df_1d, psar)
    adx_12h = align_htf_to_ltf(prices, df_1d, adx)
    
    # Volume filter: current volume > 1.5 * 20-period average
    volume_ma20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # -1: short, 0: flat, 1: long
    
    start_idx = 50  # Need enough data for indicators
    
    for i in range(start_idx, n):
        # Skip if any required data is not available
        if (np.isnan(psar_12h[i]) or 
            np.isnan(adx_12h[i]) or 
            np.isnan(volume_ma20[i])):
            signals[i] = 0.0
            continue
        
        # Volume filter: spike > 1.5x average (moderate to reduce trades)
        volume_filter = volume[i] > (1.5 * volume_ma20[i])
        
        # Trend filter: ADX > 25 indicates strong trend
        strong_trend = adx_12h[i] > 25
        
        # Price relative to Parabolic SAR
        price_above_sar = close[i] > psar_12h[i]
        price_below_sar = close[i] < psar_12h[i]
        
        if position == 0:
            # Long: Price above SAR with strong trend and volume
            if (price_above_sar and strong_trend and volume_filter):
                signals[i] = 0.25
                position = 1
            # Short: Price below SAR with strong trend and volume
            elif (price_below_sar and strong_trend and volume_filter):
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit long: Price crosses below SAR OR trend weakens
            if (close[i] < psar_12h[i]) or (adx_12h[i] < 20):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: Price crosses above SAR OR trend weakens
            if (close[i] > psar_12h[i]) or (adx_12h[i] < 20):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "12h_ADX14_1dPSAR_Volume"
timeframe = "12h"
leverage = 1.0
#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Camarilla pivot breakout with 1d volume confirmation and ADX regime filter
# - Long when price breaks above Camarilla H3 level AND ADX(14) > 25 (trending) AND volume > 1.5x 20-period average volume
# - Short when price breaks below Camarilla L3 level AND ADX(14) > 25 AND volume > 1.5x 20-period average volume
# - Exit when price crosses back inside Camarilla H3/L3 levels
# - Uses discrete position sizing 0.25 to limit fee churn
# - Target: 20-50 trades/year on 4h timeframe (80-200 total over 4 years)
# - Camarilla pivots provide mathematically derived support/resistance levels
# - ADX filter ensures we only trade in trending markets where breakouts are more reliable
# - Volume confirmation reduces false breakouts

name = "4h_1d_camarilla_adx_volume_v1"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Load HTF data ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    
    # Pre-compute 4h OHLC and volume
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Pre-compute 4h volume confirmation (20-period average)
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (1.5 * vol_ma)
    
    # Pre-compute 4h ADX(14) for regime filter
    # True Range calculation
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr1[0] = 0
    tr2[0] = 0
    tr3[0] = 0
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    
    # Directional Movement
    dm_plus = np.where((high - np.roll(high, 1)) > (np.roll(low, 1) - low), 
                       np.maximum(high - np.roll(high, 1), 0), 0)
    dm_minus = np.where((np.roll(low, 1) - low) > (high - np.roll(high, 1)), 
                        np.maximum(np.roll(low, 1) - low, 0), 0)
    dm_plus[0] = 0
    dm_minus[0] = 0
    
    # Smoothed TR, DM+ and DM- using Wilder's smoothing (alpha=1/14)
    def wilders_smoothing(arr, period):
        result = np.zeros_like(arr)
        result[period-1] = np.mean(arr[1:period+1])  # First value
        for i in range(period, len(arr)):
            result[i] = (result[i-1] * (period-1) + arr[i]) / period
        return result
    
    atr_4h = wilders_smoothing(tr, 14)
    dm_plus_smooth = wilders_smoothing(dm_plus, 14)
    dm_minus_smooth = wilders_smoothing(dm_minus, 14)
    
    # DI+ and DI-
    di_plus = 100 * dm_plus_smooth / np.where(atr_4h == 0, 1, atr_4h)
    di_minus = 100 * dm_minus_smooth / np.where(atr_4h == 0, 1, atr_4h)
    
    # DX and ADX
    dx = 100 * np.abs(di_plus - di_minus) / np.where((di_plus + di_minus) == 0, 1, (di_plus + di_minus))
    adx = wilders_smoothing(dx, 14)
    
    # ADX regime: trending when ADX > 25
    trending_regime = adx > 25
    
    # Pre-compute 1d Camarilla pivot levels
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Camarilla pivot calculations (based on previous day's OHLC)
    camarilla_h4 = close_1d + (high_1d - low_1d) * 1.1 / 2
    camarilla_h3 = close_1d + (high_1d - low_1d) * 1.1 / 4
    camarilla_h2 = close_1d + (high_1d - low_1d) * 1.1 / 6
    camarilla_h1 = close_1d + (high_1d - low_1d) * 1.1 / 12
    camarilla_l1 = close_1d - (high_1d - low_1d) * 1.1 / 12
    camarilla_l2 = close_1d - (high_1d - low_1d) * 1.1 / 6
    camarilla_l3 = close_1d - (high_1d - low_1d) * 1.1 / 4
    camarilla_l4 = close_1d - (high_1d - low_1d) * 1.1 / 2
    
    # Align HTF indicators to 4h timeframe
    camarilla_h3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_h3)
    camarilla_l3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_l3)
    trending_regime_aligned = align_htf_to_ltf(prices, df_1d, trending_regime)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(50, n):  # Start after warmup
        # Skip if any required data is invalid
        if (np.isnan(vol_ma[i]) or np.isnan(camarilla_h3_aligned[i]) or 
            np.isnan(camarilla_l3_aligned[i]) or np.isnan(trending_regime_aligned[i])):
            if position == 0:
                signals[i] = 0.0
            elif position == 1:
                signals[i] = 0.25
            else:
                signals[i] = -0.25
            continue
        
        if position == 0:  # Flat - look for new entries
            # Long conditions: price breaks above Camarilla H3 AND trending regime AND volume spike
            if (close[i] > camarilla_h3_aligned[i] and 
                trending_regime_aligned[i] and 
                volume_spike[i]):
                position = 1
                signals[i] = 0.25
            # Short conditions: price breaks below Camarilla L3 AND trending regime AND volume spike
            elif (close[i] < camarilla_l3_aligned[i] and 
                  trending_regime_aligned[i] and 
                  volume_spike[i]):
                position = -1
                signals[i] = -0.25
            else:
                signals[i] = 0.0
        else:  # Have position - look for exit
            # Exit conditions: price crosses back inside Camarilla H3/L3 levels
            exit_long = (position == 1 and close[i] < camarilla_h3_aligned[i])
            exit_short = (position == -1 and close[i] > camarilla_l3_aligned[i])
            
            if exit_long or exit_short:
                position = 0
                signals[i] = 0.0
            else:
                if position == 1:
                    signals[i] = 0.25
                else:
                    signals[i] = -0.25
    
    return signals
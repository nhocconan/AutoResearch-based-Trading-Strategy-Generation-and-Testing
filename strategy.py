#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Camarilla pivot breakout with 1d ADX trend filter and volume confirmation
# - Long when price breaks above Camarilla H3 AND 1d ADX > 25 (trending regime) AND volume > 2.0x 20-period average
# - Short when price breaks below Camarilla L3 AND 1d ADX > 25 AND volume > 2.0x 20-period average
# - Exit when price returns to Camarilla PIVOT level (mean reversion to equilibrium)
# - Uses discrete position sizing 0.25 to limit fee churn
# - Target: 19-50 trades/year on 4h timeframe (75-200 total over 4 years)
# - Camarilla levels provide institutional support/resistance that work in both trending and ranging markets
# - ADX filter ensures we only trade during strong trends when breakouts are more reliable
# - Volume confirmation reduces false breakouts
# - Works in both bull and bear markets as it follows institutional levels with trend confirmation

name = "4h_1d_camarilla_adx_volume_v1"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Load HTF data ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:  # Need enough data for ADX calculation
        return np.zeros(n)
    
    # Pre-compute 4h OHLC and volume
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Pre-compute 4h volume confirmation (20-period average)
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (2.0 * vol_ma)
    
    # Pre-compute 1d ADX(14) for trend filter
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # True Range calculation
    tr1 = high_1d - low_1d
    tr2 = np.abs(high_1d - np.roll(close_1d, 1))
    tr3 = np.abs(low_1d - np.roll(close_1d, 1))
    tr1[0] = 0
    tr2[0] = 0
    tr3[0] = 0
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    
    # Directional Movement
    dm_plus = np.where((high_1d - np.roll(high_1d, 1)) > (np.roll(low_1d, 1) - low_1d), 
                       np.maximum(high_1d - np.roll(high_1d, 1), 0), 0)
    dm_minus = np.where((np.roll(low_1d, 1) - low_1d) > (high_1d - np.roll(high_1d, 1)), 
                        np.maximum(np.roll(low_1d, 1) - low_1d, 0), 0)
    dm_plus[0] = 0
    dm_minus[0] = 0
    
    # Smoothed TR, DM+ and DM- using Wilder's smoothing (alpha=1/14)
    tr_14 = np.zeros_like(tr)
    dm_plus_14 = np.zeros_like(tr)
    dm_minus_14 = np.zeros_like(tr)
    
    # Initial values (simple average of first 14 periods)
    tr_14[13] = np.mean(tr[1:15])
    dm_plus_14[13] = np.mean(dm_plus[1:15])
    dm_minus_14[13] = np.mean(dm_minus[1:15])
    
    # Wilder's smoothing for remaining periods
    for i in range(15, len(tr)):
        tr_14[i] = (tr_14[i-1] * 13 + tr[i]) / 14
        dm_plus_14[i] = (dm_plus_14[i-1] * 13 + dm_plus[i]) / 14
        dm_minus_14[i] = (dm_minus_14[i-1] * 13 + dm_minus[i]) / 14
    
    # Directional Indicators
    di_plus = np.where(tr_14 > 0, (dm_plus_14 / tr_14) * 100, 0)
    di_minus = np.where(tr_14 > 0, (dm_minus_14 / tr_14) * 100, 0)
    
    # DX and ADX
    dx = np.where((di_plus + di_minus) > 0, 
                  np.abs(di_plus - di_minus) / (di_plus + di_minus) * 100, 0)
    
    # ADX using Wilder's smoothing
    adx = np.zeros_like(dx)
    adx[27] = np.mean(dx[14:28])  # First ADX value after 2*14 periods
    for i in range(28, len(dx)):
        adx[i] = (adx[i-1] * 13 + dx[i]) / 14
    
    # Trend regime: ADX > 25 indicates strong trend
    strong_trend = adx > 25
    
    # Pre-compute 4h Camarilla levels from previous period's OHLC
    prev_high = np.roll(high, 1)
    prev_low = np.roll(low, 1)
    prev_close = np.roll(close, 1)
    prev_high[0] = high[0]
    prev_low[0] = low[0]
    prev_close[0] = close[0]
    
    # Calculate pivot and ranges
    pivot = (prev_high + prev_low + prev_close) / 3
    range_hl = prev_high - prev_low
    
    # Camarilla levels
    camarilla_h3 = pivot + (range_hl * 1.1 / 4)
    camarilla_l3 = pivot - (range_hl * 1.1 / 4)
    camarilla_h4 = pivot + (range_hl * 1.1 / 2)
    camarilla_l4 = pivot - (range_hl * 1.1 / 2)
    
    # Align HTF indicators to 4h timeframe
    strong_trend_aligned = align_htf_to_ltf(prices, df_1d, strong_trend)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(50, n):  # Start after warmup
        # Skip if any required data is invalid
        if (np.isnan(pivot[i]) or np.isnan(camarilla_h3[i]) or np.isnan(camarilla_l3[i]) or 
            np.isnan(vol_ma[i]) or np.isnan(strong_trend_aligned[i])):
            if position == 0:
                signals[i] = 0.0
            elif position == 1:
                signals[i] = 0.25
            else:
                signals[i] = -0.25
            continue
        
        if position == 0:  # Flat - look for new entries
            # Long conditions: price breaks above Camarilla H3 AND strong trend AND volume spike
            if (close[i] > camarilla_h3[i] and 
                strong_trend_aligned[i] and 
                volume_spike[i]):
                position = 1
                signals[i] = 0.25
            # Short conditions: price breaks below Camarilla L3 AND strong trend AND volume spike
            elif (close[i] < camarilla_l3[i] and 
                  strong_trend_aligned[i] and 
                  volume_spike[i]):
                position = -1
                signals[i] = -0.25
            else:
                signals[i] = 0.0
        else:  # Have position - look for exit to pivot (mean reversion)
            # Exit when price returns to pivot level (mean reversion to equilibrium)
            exit_long = (position == 1 and close[i] <= pivot[i])
            exit_short = (position == -1 and close[i] >= pivot[i])
            
            if exit_long or exit_short:
                position = 0
                signals[i] = 0.0
            else:
                if position == 1:
                    signals[i] = 0.25
                else:
                    signals[i] = -0.25
    
    return signals
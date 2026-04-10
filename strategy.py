#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h Williams %R mean reversal + 1d/1w regime filter + volume confirmation
# - Long when Williams %R(14) < -80 (oversold) AND 1d ADX < 25 (low trend regime) AND 1w close > 1w EMA50 (bullish weekly bias) AND volume > 1.5x 20-period average
# - Short when Williams %R(14) > -20 (overbought) AND 1d ADX < 25 AND 1w close < 1w EMA50 AND volume > 1.5x 20-period average
# - Exit when Williams %R returns to -50 (mean reversion to midpoint)
# - Uses discrete position sizing 0.25 to limit fee churn
# - Target: 12-37 trades/year on 12h timeframe (50-150 total over 4 years)
# - Williams %R identifies overextended moves likely to reverse
# - ADX filter ensures we trade in ranging markets where mean reversion works
# - Weekly EMA filter aligns with higher timeframe trend to avoid counter-trend trades
# - Volume confirmation reduces false signals

name = "12h_1d_1w_williamsr_regime_volume_v1"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Load HTF data ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    df_1w = get_htf_data(prices, '1w')
    if len(df_1d) < 30 or len(df_1w) < 50:
        return np.zeros(n)
    
    # Pre-compute 12h OHLC and volume
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Pre-compute 12h volume confirmation (20-period average)
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (1.5 * vol_ma)
    
    # Pre-compute 12h Williams %R(14)
    highest_high = pd.Series(high).rolling(window=14, min_periods=14).max().values
    lowest_low = pd.Series(low).rolling(window=14, min_periods=14).min().values
    williams_r = -100 * (highest_high - close) / (highest_high - lowest_low)
    williams_r[highest_high == lowest_low] = -50  # Avoid division by zero
    
    # Pre-compute 1d ADX(14) for regime filter
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # True Range and Directional Movement
    tr1 = high_1d - low_1d
    tr2 = np.abs(high_1d - np.roll(close_1d, 1))
    tr3 = np.abs(low_1d - np.roll(close_1d, 1))
    tr1[0] = 0
    tr2[0] = 0
    tr3[0] = 0
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    
    dm_plus = np.where((high_1d - np.roll(high_1d, 1)) > (np.roll(low_1d, 1) - low_1d), 
                       np.maximum(high_1d - np.roll(high_1d, 1), 0), 0)
    dm_minus = np.where((np.roll(low_1d, 1) - low_1d) > (high_1d - np.roll(high_1d, 1)), 
                        np.maximum(np.roll(low_1d, 1) - low_1d, 0), 0)
    dm_plus[0] = 0
    dm_minus[0] = 0
    
    # Smoothed TR, DM+, DM- using Wilder's smoothing (alpha=1/14)
    tr_14 = np.zeros_like(tr)
    dm_plus_14 = np.zeros_like(dm_plus)
    dm_minus_14 = np.zeros_like(dm_minus)
    tr_14[13] = np.mean(tr[1:14])
    dm_plus_14[13] = np.mean(dm_plus[1:14])
    dm_minus_14[13] = np.mean(dm_minus[1:14])
    for i in range(14, len(tr)):
        tr_14[i] = (tr_14[i-1] * 13 + tr[i]) / 14
        dm_plus_14[i] = (dm_plus_14[i-1] * 13 + dm_plus[i]) / 14
        dm_minus_14[i] = (dm_minus_14[i-1] * 13 + dm_minus[i]) / 14
    
    # DI+ and DI-
    di_plus = 100 * dm_plus_14 / tr_14
    di_minus = 100 * dm_minus_14 / tr_14
    dx = 100 * np.abs(di_plus - di_minus) / (di_plus + di_minus)
    dx[di_plus + di_minus == 0] = 0
    
    # ADX(14) - smoothed DX
    adx = np.zeros_like(dx)
    adx[27] = np.mean(dx[14:28])  # First ADX value after DX period
    for i in range(28, len(dx)):
        adx[i] = (adx[i-1] * 13 + dx[i]) / 14
    
    # Regime: low trend when ADX < 25
    low_trend_regime = adx < 25
    
    # Pre-compute 1w EMA50 for trend filter
    close_1w = df_1w['close'].values
    ema_50_1w = pd.Series(close_1w).ewm(span=50, adjust=False, min_periods=50).mean().values
    weekly_bullish = close_1w > ema_50_1w
    weekly_bearish = close_1w < ema_50_1w
    
    # Align HTF indicators to 12h timeframe
    low_trend_regime_aligned = align_htf_to_ltf(prices, df_1d, low_trend_regime)
    weekly_bullish_aligned = align_htf_to_ltf(prices, df_1w, weekly_bullish)
    weekly_bearish_aligned = align_htf_to_ltf(prices, df_1w, weekly_bearish)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(100, n):  # Start after warmup
        # Skip if any required data is invalid
        if (np.isnan(williams_r[i]) or np.isnan(vol_ma[i]) or 
            np.isnan(low_trend_regime_aligned[i]) or 
            np.isnan(weekly_bullish_aligned[i]) or np.isnan(weekly_bearish_aligned[i])):
            if position == 0:
                signals[i] = 0.0
            elif position == 1:
                signals[i] = 0.25
            else:
                signals[i] = -0.25
            continue
        
        if position == 0:  # Flat - look for new entries
            # Long conditions: Williams %R oversold AND low trend regime AND weekly bullish AND volume spike
            if (williams_r[i] < -80 and 
                low_trend_regime_aligned[i] and 
                weekly_bullish_aligned[i] and 
                volume_spike[i]):
                position = 1
                signals[i] = 0.25
            # Short conditions: Williams %R overbought AND low trend regime AND weekly bearish AND volume spike
            elif (williams_r[i] > -20 and 
                  low_trend_regime_aligned[i] and 
                  weekly_bearish_aligned[i] and 
                  volume_spike[i]):
                position = -1
                signals[i] = -0.25
            else:
                signals[i] = 0.0
        else:  # Have position - look for exit to Williams %R = -50 (mean reversion)
            # Exit when Williams %R returns to -50 (mean reversion to midpoint)
            exit_long = (position == 1 and williams_r[i] >= -50)
            exit_short = (position == -1 and williams_r[i] <= -50)
            
            if exit_long or exit_short:
                position = 0
                signals[i] = 0.0
            else:
                if position == 1:
                    signals[i] = 0.25
                else:
                    signals[i] = -0.25
    
    return signals
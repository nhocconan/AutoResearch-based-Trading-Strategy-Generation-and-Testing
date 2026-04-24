#!/usr/bin/env python3
"""
Hypothesis: 1h RSI(14) mean reversion with 4h ADX(14) regime filter and volume confirmation.
- Primary timeframe: 1h for entry timing.
- HTF: 4h ADX(14) to filter regime (ADX > 25 = trending, avoid entries; ADX <= 25 = ranging, mean revert).
- Volume: Current 1h volume > 1.5 * 20-period volume MA to confirm participation.
- Entry: Long when RSI(14) < 30 AND ADX <= 25 AND volume spike.
         Short when RSI(14) > 70 AND ADX <= 25 AND volume spike.
- Exit: RSI returns to neutral zone (40 < RSI < 60) or loss of volume confirmation.
- Signal size: 0.20 discrete to limit drawdown and reduce fee churn.
- Session filter: Trade only between 08:00-20:00 UTC to avoid low-liquidity hours.
- Target: 60-150 total trades over 4 years (15-37/year) for 1h timeframe.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Extract price and volume data
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Pre-compute session filter (08:00-20:00 UTC)
    hours = prices.index.hour
    in_session = (hours >= 8) & (hours <= 20)
    
    # Calculate RSI(14) on 1h
    delta = pd.Series(close).diff()
    gain = delta.where(delta > 0, 0.0)
    loss = (-delta).where(delta < 0, 0.0)
    avg_gain = gain.ewm(alpha=1/14, adjust=False, min_periods=14).mean()
    avg_loss = loss.ewm(alpha=1/14, adjust=False, min_periods=14).mean()
    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    rsi_values = rsi.values
    
    # Get 4h data for ADX(14) and volume MA
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 50:
        return np.zeros(n)
    
    # Calculate ADX(14) on 4h
    high_4h = df_4h['high'].values
    low_4h = df_4h['low'].values
    close_4h = df_4h['close'].values
    
    # True Range
    tr1 = high_4h - low_4h
    tr2 = np.abs(high_4h - np.roll(close_4h, 1))
    tr3 = np.abs(low_4h - np.roll(close_4h, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]  # First period
    
    # Directional Movement
    dm_plus = np.where((high_4h - np.roll(high_4h, 1)) > (np.roll(low_4h, 1) - low_4h),
                       np.maximum(high_4h - np.roll(high_4h, 1), 0), 0)
    dm_minus = np.where((np.roll(low_4h, 1) - low_4h) > (high_4h - np.roll(high_4h, 1)),
                        np.maximum(np.roll(low_4h, 1) - low_4h, 0), 0)
    dm_plus[0] = 0
    dm_minus[0] = 0
    
    # Smooth TR, DM+, DM- with Wilder's smoothing (alpha=1/14)
    tr_14 = pd.Series(tr).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    dm_plus_14 = pd.Series(dm_plus).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    dm_minus_14 = pd.Series(dm_minus).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    
    # DI+ and DI-
    di_plus = 100 * dm_plus_14 / tr_14
    di_minus = 100 * dm_minus_14 / tr_14
    
    # DX and ADX
    dx = 100 * np.abs(di_plus - di_minus) / (di_plus + di_minus)
    adx = pd.Series(dx).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    
    # Calculate 20-period volume MA on 4h
    vol_ma_4h = pd.Series(df_4h['volume'].values).rolling(window=20, min_periods=20).mean().values
    
    # Align HTF indicators to 1h
    adx_aligned = align_htf_to_ltf(prices, df_4h, adx)
    vol_ma_4h_aligned = align_htf_to_ltf(prices, df_4h, vol_ma_4h)
    
    # Volume confirmation: current 1h volume > 1.5 * 20-period 4h volume MA (aligned)
    volume_spike = volume > (1.5 * vol_ma_4h_aligned)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start from index where all indicators are ready
    start_idx = max(50, 30)  # Need enough 4h bars for ADX and volume MA
    
    for i in range(start_idx, n):
        # Skip if data not ready or outside session
        if (np.isnan(adx_aligned[i]) or np.isnan(rsi_values[i]) or 
            np.isnan(volume_spike[i]) or not in_session[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        adx_val = adx_aligned[i]
        rsi_val = rsi_values[i]
        vol_spike = volume_spike[i]
        
        if position == 0:
            # Check for entry signals in ranging market (ADX <= 25) with volume spike
            if vol_spike and adx_val <= 25:
                # Oversold: RSI < 30
                if rsi_val < 30:
                    signals[i] = 0.20
                    position = 1
                # Overbought: RSI > 70
                elif rsi_val > 70:
                    signals[i] = -0.20
                    position = -1
        elif position == 1:
            # Long exit: RSI returns to neutral (40 < RSI < 60) or loss of volume confirmation
            if rsi_val > 40 and rsi_val < 60 or not vol_spike:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.20
        elif position == -1:
            # Short exit: RSI returns to neutral (40 < RSI < 60) or loss of volume confirmation
            if rsi_val > 40 and rsi_val < 60 or not vol_spike:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.20
    
    return signals

name = "1h_RSI14_4hADX25_VolumeSpike_Session_v1"
timeframe = "1h"
leverage = 1.0
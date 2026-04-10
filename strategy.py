#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1h Camarilla pivot breakout with 4h volume confirmation and 1d ADX trend filter
# - Camarilla levels from 4h: R3/S3 for breakout continuation (strong momentum)
# - 1d ADX(14) > 20 to ensure daily trend alignment and avoid choppy markets
# - Volume confirmation: current 1h volume > 1.5x 20-period average to confirm institutional interest
# - Designed for 1h timeframe: targets 15-37 trades/year (60-150 total over 4 years) to avoid fee drag
# - Uses discrete position sizing (0.20) to minimize fee churn
# - ATR-based stoploss: exit when price moves against position by 2.0x ATR(14)

name = "1h_4h_1d_camarilla_adx_volume_atr_v1"
timeframe = "1h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Load HTF data ONCE before loop
    df_4h = get_htf_data(prices, '4h')
    df_1d = get_htf_data(prices, '1d')
    if len(df_4h) < 20 or len(df_1d) < 20:
        return np.zeros(n)
    
    # Pre-compute 1d ADX(14) for trend filter
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # True Range
    tr1 = high_1d - low_1d
    tr2 = np.abs(high_1d - np.roll(close_1d, 1))
    tr3 = np.abs(low_1d - np.roll(close_1d, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]
    
    # Directional Movement
    dm_plus = np.where((high_1d - np.roll(high_1d, 1)) > (np.roll(low_1d, 1) - low_1d), 
                       np.maximum(high_1d - np.roll(high_1d, 1), 0), 0)
    dm_minus = np.where((np.roll(low_1d, 1) - low_1d) > (high_1d - np.roll(high_1d, 1)), 
                        np.maximum(np.roll(low_1d, 1) - low_1d, 0), 0)
    dm_plus[0] = 0
    dm_minus[0] = 0
    
    # Smoothed values
    tr_14 = pd.Series(tr).ewm(span=14, adjust=False, min_periods=14).mean().values
    dm_plus_14 = pd.Series(dm_plus).ewm(span=14, adjust=False, min_periods=14).mean().values
    dm_minus_14 = pd.Series(dm_minus).ewm(span=14, adjust=False, min_periods=14).mean().values
    
    # Directional Indicators
    di_plus = 100 * dm_plus_14 / tr_14
    di_minus = 100 * dm_minus_14 / tr_14
    
    # DX and ADX
    dx = 100 * np.abs(di_plus - di_minus) / (di_plus + di_minus + 1e-10)
    adx = pd.Series(dx).ewm(span=14, adjust=False, min_periods=14).mean().values
    adx_aligned = align_htf_to_ltf(prices, df_1d, adx)
    
    # Pre-compute 4h Camarilla levels
    high_4h = df_4h['high'].values
    low_4h = df_4h['low'].values
    close_4h = df_4h['close'].values
    
    # Camarilla levels calculation
    rango = high_4h - low_4h
    camarilla_r3 = close_4h + (rango * 1.1 / 4)
    camarilla_s3 = close_4h - (rango * 1.1 / 4)
    
    camarilla_r3_aligned = align_htf_to_ltf(prices, df_4h, camarilla_r3)
    camarilla_s3_aligned = align_htf_to_ltf(prices, df_4h, camarilla_s3)
    
    # Pre-compute 1h ATR(14) for stoploss
    high_1h = prices['high'].values
    low_1h = prices['low'].values
    close_1h = prices['close'].values
    
    # True Range for 1h
    tr1_1h = high_1h - low_1h
    tr2_1h = np.abs(high_1h - np.roll(close_1h, 1))
    tr3_1h = np.abs(low_1h - np.roll(close_1h, 1))
    tr_1h = np.maximum(tr1_1h, np.maximum(tr2_1h, tr3_1h))
    tr_1h[0] = tr1_1h[0]
    
    atr_14 = pd.Series(tr_1h).ewm(span=14, adjust=False, min_periods=14).mean().values
    
    # Pre-compute 1h volume confirmation
    volume_1h = prices['volume'].values
    avg_volume_20 = pd.Series(volume_1h).rolling(window=20, min_periods=20).mean().values
    vol_spike = volume_1h > (1.5 * avg_volume_20)
    
    # Session filter: 08-20 UTC
    hours = prices.index.hour
    in_session = (hours >= 8) & (hours <= 20)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    entry_price = 0.0
    
    for i in range(50, n):
        # Skip if any required data is invalid or outside session
        if (np.isnan(adx_aligned[i]) or np.isnan(camarilla_r3_aligned[i]) or 
            np.isnan(camarilla_s3_aligned[i]) or np.isnan(vol_spike[i]) or
            np.isnan(atr_14[i]) or not in_session[i]):
            signals[i] = 0.0
            continue
        
        if position == 1:  # Long position
            # Exit: ATR-based stoploss or price closes below S3 (mean reversion failure)
            if prices['close'].iloc[i] < entry_price - 2.0 * atr_14[i] or prices['close'].iloc[i] < camarilla_s3_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.20
                
        elif position == -1:  # Short position
            # Exit: ATR-based stoploss or price closes above R3 (mean reversion failure)
            if prices['close'].iloc[i] > entry_price + 2.0 * atr_14[i] or prices['close'].iloc[i] > camarilla_r3_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.20
        else:  # Flat
            # Look for Camarilla breakout with trend and volume filters
            if vol_spike[i] and adx_aligned[i] > 20 and in_session[i]:
                # Breakout long: price closes above R3 (bullish breakout)
                if prices['close'].iloc[i] > camarilla_r3_aligned[i]:
                    position = 1
                    entry_price = prices['close'].iloc[i]
                    signals[i] = 0.20
                # Breakout short: price closes below S3 (bearish breakout)
                elif prices['close'].iloc[i] < camarilla_s3_aligned[i]:
                    position = -1
                    entry_price = prices['close'].iloc[i]
                    signals[i] = -0.20
    
    return signals
#!/usr/bin/env python3
"""
4h_Camarilla_R3_S3_Breakout_1dEMA34_RegimeFilter_v1
Hypothesis: Camarilla R3/S3 breakout with 1d EMA34 trend and ADX regime filter avoids whipsaws in ranging markets. Only trades in trending regimes (ADX > 25) to improve win rate and reduce false breakouts. Volume confirmation ensures institutional participation. Designed for BTC/ETH: trend filter works in bull/bear, volume confirms breakout validity, ADX filter prevents overtrading in chop. Target 15-30 trades/year to minimize fee drag.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Load 1d data ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 35:  # Need enough for EMA34 and ADX
        return np.zeros(n)
    
    # 1d EMA34 for trend filter
    ema_34_1d = pd.Series(df_1d['close']).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # 1d ADX(14) for regime filter
    # Calculate True Range
    tr1 = df_1d['high'] - df_1d['low']
    tr2 = np.abs(df_1d['high'] - df_1d['close'].shift(1))
    tr3 = np.abs(df_1d['low'] - df_1d['close'].shift(1))
    tr_1d = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1).values
    tr_1d[0] = 0  # First value has no previous close
    
    # Calculate +DM and -DM
    up_move = df_1d['high'].diff().values
    down_move = df_1d['low'].diff().values * -1  # Positive when low decreases
    up_move[0] = 0
    down_move[0] = 0
    
    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0)
    
    # Smooth TR, +DM, -DM with Wilder's smoothing (alpha = 1/period)
    def wilder_smooth(data, period):
        result = np.full_like(data, np.nan)
        if len(data) < period:
            return result
        # First value is simple average
        result[period-1] = np.mean(data[:period])
        # Subsequent values: Wilder's smoothing
        for i in range(period, len(data)):
            result[i] = result[i-1] - (result[i-1] / period) + data[i]
        return result
    
    atr_1d = wilder_smooth(tr_1d, 14)
    plus_di_1d = 100 * wilder_smooth(plus_dm, 14) / atr_1d
    minus_di_1d = 100 * wilder_smooth(minus_dm, 14) / atr_1d
    dx_1d = 100 * np.abs(plus_di_1d - minus_di_1d) / (plus_di_1d + minus_di_1d)
    adx_1d = wilder_smooth(dx_1d, 14)
    
    # Align to 4h (wait for completed 1d bar)
    ema_34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    adx_1d_aligned = align_htf_to_ltf(prices, df_1d, adx_1d)
    
    # Calculate Camarilla levels from previous 1d bar
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    camarilla_r3 = close_1d + (high_1d - low_1d) * 1.1 / 4
    camarilla_s3 = close_1d - (high_1d - low_1d) * 1.1 / 4
    
    # Align to 4h (wait for completed 1d bar)
    camarilla_r3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_r3)
    camarilla_s3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_s3)
    
    # Average volume for confirmation (20-period SMA)
    avg_volume = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    base_size = 0.25
    
    # Warmup: max of EMA(34), volume(20), ADX(14+14+14)
    start_idx = max(34, 20, 42)  # EMA34, volume20, ADX needs 14+14+14=42
    
    for i in range(start_idx, n):
        close_val = close[i]
        vol = volume[i]
        avg_vol = avg_volume[i]
        ema_val = ema_34_1d_aligned[i]
        adx_val = adx_1d_aligned[i]
        r3_val = camarilla_r3_aligned[i]
        s3_val = camarilla_s3_aligned[i]
        
        # Skip if any data not ready
        if (np.isnan(ema_val) or np.isnan(avg_vol) or np.isnan(adx_val) or 
            np.isnan(r3_val) or np.isnan(s3_val)):
            # Hold current position
            signals[i] = base_size if position == 1 else (-base_size if position == -1 else 0.0)
            continue
        
        # Volume confirmation: current volume > 1.5x average volume (slightly relaxed for more trades)
        volume_confirmed = vol > 1.5 * avg_vol
        
        # Regime filter: only trade in trending markets (ADX > 25)
        trending_regime = adx_val > 25
        
        # Long: price CLOSES above R3 with 1d uptrend, volume, and trending regime
        long_condition = (close_val > r3_val) and (close_val > ema_val) and volume_confirmed and trending_regime
        # Short: price CLOSES below S3 with 1d downtrend, volume, and trending regime
        short_condition = (close_val < s3_val) and (close_val < ema_val) and volume_confirmed and trending_regime
        
        # Exit: price retests broken level OR reverse signal
        long_exit = (position == 1 and close_val <= r3_val)
        short_exit = (position == -1 and close_val >= s3_val)
        reverse_exit_long = (position == 1 and short_condition)
        reverse_exit_short = (position == -1 and long_condition)
        
        if long_condition and position != 1:
            signals[i] = base_size
            position = 1
            entry_price = close_val
        elif short_condition and position != -1:
            signals[i] = -base_size
            position = -1
            entry_price = close_val
        elif long_exit or reverse_exit_long:
            signals[i] = 0.0
            position = 0
        elif short_exit or reverse_exit_short:
            signals[i] = 0.0
            position = 0
        else:
            # Hold position
            signals[i] = base_size if position == 1 else (-base_size if position == -1 else 0.0)
    
    return signals

name = "4h_Camarilla_R3_S3_Breakout_1dEMA34_RegimeFilter_v1"
timeframe = "4h"
leverage = 1.0
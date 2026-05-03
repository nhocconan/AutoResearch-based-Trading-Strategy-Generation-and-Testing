#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1d Camarilla R3/S3 breakout with 1w trend filter and volume confirmation.
# Long when price breaks above 1d Camarilla R3 level in 1w uptrend with volume spike (>1.5x 20-period volume MA).
# Short when price breaks below 1d Camarilla S3 level in 1w downtrend with volume spike.
# Uses ATR-based stoploss (signal→0 when price moves against position by 2.0*ATR).
# Camarilla levels provide high-probability reversal points in ranging markets, 1w EMA34 ensures higher timeframe alignment,
# Volume spike confirms institutional participation. Works in both bull and bear markets by only trading
# with the 1w trend, avoiding counter-trend whipsaws during ranging periods.

name = "1d_Camarilla_R3S3_1wEMA34_VolumeSpike_ATR"
timeframe = "1d"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    open_price = prices['open'].values
    
    # Calculate ATR for stoploss (using primary timeframe)
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr1[0] = 0
    tr2[0] = 0
    tr3[0] = 0
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    atr = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    # Get 1d data for Camarilla pivot calculation
    df_1d = get_htf_data(prices, '1d')
    
    if len(df_1d) < 2:
        return np.zeros(n)
    
    # Calculate 1d Camarilla levels (R3, S3) from previous day
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Camarilla levels: R3 = close + (high - low) * 1.1/4, S3 = close - (high - low) * 1.1/4
    camarilla_r3 = close_1d + (high_1d - low_1d) * 1.1 / 4
    camarilla_s3 = close_1d - (high_1d - low_1d) * 1.1 / 4
    
    camarilla_r3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_r3)
    camarilla_s3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_s3)
    
    # Get 1w data for trend filter
    df_1w = get_htf_data(prices, '1w')
    
    if len(df_1w) < 34:
        return np.zeros(n)
    
    # Calculate 1w EMA34
    close_1w = df_1w['close'].values
    ema_34_1w = pd.Series(close_1w).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_34_1w)
    
    # Volume spike detection (20-period volume MA on primary timeframe)
    volume_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (volume_ma * 1.5)  # Volume at least 1.5x average
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0  # Track entry price for ATR-based stoploss
    
    for i in range(100, n):
        # Skip if any value is NaN
        if (np.isnan(camarilla_r3_aligned[i]) or np.isnan(camarilla_s3_aligned[i]) or 
            np.isnan(ema_34_1w_aligned[i]) or np.isnan(volume_ma[i]) or np.isnan(atr[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
            
        close_val = close[i]
        open_val = open_price[i]
        vol_spike = volume_spike[i]
        r3_level = camarilla_r3_aligned[i]
        s3_level = camarilla_s3_aligned[i]
        trend_up = close_val > ema_34_1w_aligned[i]   # 1w uptrend
        trend_down = close_val < ema_34_1w_aligned[i]  # 1w downtrend
        
        if position == 0:
            # Long: price breaks above Camarilla R3 level AND 1w uptrend AND volume spike
            if close_val > r3_level and trend_up and vol_spike:
                signals[i] = 0.25
                position = 1
                entry_price = close_val
            # Short: price breaks below Camarilla S3 level AND 1w downtrend AND volume spike
            elif close_val < s3_level and trend_down and vol_spike:
                signals[i] = -0.25
                position = -1
                entry_price = close_val
        elif position == 1:
            # Exit conditions for long
            exit_signal = False
            # Stoploss: price moves against position by 2.0*ATR
            if close_val < entry_price - 2.0 * atr[i]:
                exit_signal = True
            # Exit: price breaks below Camarilla S3 level
            elif close_val < s3_level:
                exit_signal = True
            # Exit: 1w trend changes to downtrend
            elif not trend_up:
                exit_signal = True
            
            if exit_signal:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit conditions for short
            exit_signal = False
            # Stoploss: price moves against position by 2.0*ATR
            if close_val > entry_price + 2.0 * atr[i]:
                exit_signal = True
            # Exit: price breaks above Camarilla R3 level
            elif close_val > r3_level:
                exit_signal = True
            # Exit: 1w trend changes to uptrend
            elif not trend_down:
                exit_signal = True
            
            if exit_signal:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals
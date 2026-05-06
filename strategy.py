#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Camarilla R3/S3 breakout with 1d EMA34 trend filter and volume confirmation
# Long when price breaks above R3 AND price > 1d EMA34 (uptrend) AND volume > 1.5 * 20-period avg volume
# Short when price breaks below S3 AND price < 1d EMA34 (downtrend) AND volume > 1.5 * 20-period avg volume
# Exit with ATR-based trailing stop: signal→0 when long and price < highest_high - 2.0 * ATR OR short and price > lowest_low + 2.0 * ATR
# Uses discrete sizing 0.25 to manage drawdown (BTC -77% in 2022 → ~19.25% loss at 0.25 exposure)
# Target: 50-150 total trades over 4 years (12-37/year) for 6h timeframe
# Camarilla pivots provide mathematical support/resistance levels based on prior day's range
# Volume confirmation ensures conviction behind breakouts, reducing false signals
# ATR trailing stop manages risk while allowing trends to develop
# Works in bull via buying breakouts in uptrend, works in bear via selling breakdowns in downtrend

name = "6h_Camarilla_R3S3_1dEMA34_Volume_v2"
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
    
    # Get 1d data ONCE before loop for EMA34 trend filter and Camarilla calculation
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 34:
        return np.zeros(n)
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate 1d EMA34
    close_1d_series = pd.Series(close_1d)
    ema_34_1d = close_1d_series.ewm(span=34, adjust=False, min_periods=34).mean().values
    
    # Calculate Camarilla pivot levels for 1d: based on prior day's high, low, close
    # R4 = close + (high - low) * 1.1/2
    # R3 = close + (high - low) * 1.1/4
    # R2 = close + (high - low) * 1.1/6
    # R1 = close + (high - low) * 1.1/12
    # PP = (high + low + close) / 3
    # S1 = close - (high - low) * 1.1/12
    # S2 = close - (high - low) * 1.1/6
    # S3 = close - (high - low) * 1.1/4
    # S4 = close - (high - low) * 1.1/2
    camarilla_r3 = close_1d + (high_1d - low_1d) * 1.1 / 4
    camarilla_s3 = close_1d - (high_1d - low_1d) * 1.1 / 4
    
    # Calculate 20-period ATR(20) for stoploss
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]  # First period TR is just high-low
    atr_20 = pd.Series(tr).ewm(span=20, adjust=False, min_periods=20).mean().values
    
    # Calculate volume confirmation: volume > 1.5 * 20-period average volume
    avg_volume_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (1.5 * avg_volume_20)
    
    # Align HTF indicators to 6h timeframe (wait for completed HTF bar)
    # We want to use the PREVIOUS completed 1d bar's values for the current 6h bar
    # So we shift the 1d indicators by 1 bar before aligning
    ema_34_1d_lagged = np.roll(ema_34_1d, 1)
    ema_34_1d_lagged[0] = np.nan  # First value becomes NaN after shift
    camarilla_r3_lagged = np.roll(camarilla_r3, 1)
    camarilla_r3_lagged[0] = np.nan
    camarilla_s3_lagged = np.roll(camarilla_s3, 1)
    camarilla_s3_lagged[0] = np.nan
    
    ema_34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d_lagged)
    camarilla_r3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_r3_lagged)
    camarilla_s3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_s3_lagged)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    highest_high_since_entry = 0.0
    lowest_low_since_entry = 0.0
    
    for i in range(50, n):  # Start after warmup period
        # Skip if any value is NaN
        if (np.isnan(ema_34_1d_aligned[i]) or np.isnan(camarilla_r3_aligned[i]) or np.isnan(camarilla_s3_aligned[i]) or
            np.isnan(atr_20[i]) or np.isnan(volume_spike[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
                highest_high_since_entry = 0.0
                lowest_low_since_entry = 0.0
            continue
        
        if position == 0:
            # Camarilla breakout signals with trend and volume filters
            breakout_up = close[i] > camarilla_r3_aligned[i]  # Price breaks above R3
            breakout_down = close[i] < camarilla_s3_aligned[i]  # Price breaks below S3
            
            # Long: breakout up AND uptrend AND volume spike
            if breakout_up and close[i] > ema_34_1d_aligned[i] and volume_spike[i]:
                signals[i] = 0.25
                position = 1
                highest_high_since_entry = high[i]
            # Short: breakout down AND downtrend AND volume spike
            elif breakout_down and close[i] < ema_34_1d_aligned[i] and volume_spike[i]:
                signals[i] = -0.25
                position = -1
                lowest_low_since_entry = low[i]
        elif position == 1:
            # Update highest high since entry
            highest_high_since_entry = max(highest_high_since_entry, high[i])
            # Exit long: price drops below highest_high - 2.0 * ATR (trailing stop)
            if close[i] < highest_high_since_entry - 2.0 * atr_20[i]:
                signals[i] = 0.0
                position = 0
                highest_high_since_entry = 0.0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Update lowest low since entry
            lowest_low_since_entry = min(lowest_low_since_entry, low[i])
            # Exit short: price rises above lowest_low + 2.0 * ATR (trailing stop)
            if close[i] > lowest_low_since_entry + 2.0 * atr_20[i]:
                signals[i] = 0.0
                position = 0
                lowest_low_since_entry = 0.0
            else:
                signals[i] = -0.25
    
    return signals
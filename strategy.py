#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1h Camarilla R1/S1 breakout with 4h EMA50 trend filter and volume spike
# Long when price breaks above R1 AND price > 4h EMA50 (uptrend) AND volume > 2.0 * 20-bar avg volume
# Short when price breaks below S1 AND price < 4h EMA50 (downtrend) AND volume > 2.0 * 20-bar avg volume
# Exit with ATR-based trailing stop: signal→0 when long and price < highest_high - 2.0 * ATR OR short and price > lowest_low + 2.0 * ATR
# Uses discrete sizing 0.20 to manage drawdown (BTC -77% in 2022 → ~15.4% loss at 0.20 exposure)
# Target: 60-150 total trades over 4 years (15-37/year) for 1h timeframe
# Uses 4h/1d for signal direction, 1h only for entry timing
# Session filter: 08-20 UTC to reduce noise trades
# Camarilla pivots provide mathematically derived support/resistance levels with institutional relevance
# Volume confirmation ensures conviction behind breakouts, reducing false signals
# ATR trailing stop manages risk while allowing trends to develop

name = "1h_Camarilla_R1S1_4hEMA50_VolumeSpike_Session_v1"
timeframe = "1h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Precompute session hours (08-20 UTC) ONCE before loop
    hours = pd.DatetimeIndex(prices['open_time']).hour
    in_session = (hours >= 8) & (hours <= 20)
    
    # Get 4h data ONCE before loop for EMA50 trend filter
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 50:
        return np.zeros(n)
    close_4h = df_4h['close'].values
    
    # Calculate 4h EMA50
    close_4h_series = pd.Series(close_4h)
    ema_50_4h = close_4h_series.ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Calculate Camarilla pivot levels for 1h timeframe using previous 1d OHLC
    # Camarilla: R1 = close + 0.105*(high-low), S1 = close - 0.105*(high-low)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    prev_close_1d = np.roll(df_1d['close'].values, 1)
    prev_high_1d = np.roll(df_1d['high'].values, 1)
    prev_low_1d = np.roll(df_1d['low'].values, 1)
    prev_close_1d[0] = np.nan
    prev_high_1d[0] = np.nan
    prev_low_1d[0] = np.nan
    
    # Calculate Camarilla levels for previous day
    camarilla_R1 = prev_close_1d + 0.105 * (prev_high_1d - prev_low_1d)
    camarilla_S1 = prev_close_1d - 0.105 * (prev_high_1d - prev_low_1d)
    
    # Align HTF indicators to 1h timeframe (wait for completed HTF bar)
    ema_50_4h_aligned = align_htf_to_ltf(prices, df_4h, ema_50_4h)
    camarilla_R1_aligned = align_htf_to_ltf(prices, df_1d, camarilla_R1)
    camarilla_S1_aligned = align_htf_to_ltf(prices, df_1d, camarilla_S1)
    
    # Calculate 20-period ATR(20) for stoploss on 1h data
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]  # First period TR is just high-low
    atr_20 = pd.Series(tr).ewm(span=20, adjust=False, min_periods=20).mean().values
    
    # Calculate volume confirmation: volume > 2.0 * 20-bar average volume
    avg_volume_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (2.0 * avg_volume_20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    highest_high_since_entry = 0.0
    lowest_low_since_entry = 0.0
    
    for i in range(50, n):  # Start after warmup period
        # Skip if any value is NaN
        if (np.isnan(ema_50_4h_aligned[i]) or np.isnan(camarilla_R1_aligned[i]) or np.isnan(camarilla_S1_aligned[i]) or
            np.isnan(atr_20[i]) or np.isnan(volume_spike[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
                highest_high_since_entry = 0.0
                lowest_low_since_entry = 0.0
            continue
        
        # Session filter: only trade between 08-20 UTC
        if not in_session[i]:
            if position != 0:
                signals[i] = 0.0
                position = 0
                highest_high_since_entry = 0.0
                lowest_low_since_entry = 0.0
            continue
        
        if position == 0:
            # Camarilla breakout signals with trend and volume filters
            breakout_up = close[i] > camarilla_R1_aligned[i]  # Price breaks above R1
            breakout_down = close[i] < camarilla_S1_aligned[i]  # Price breaks below S1
            
            # Long: breakout up AND uptrend AND volume spike
            if breakout_up and close[i] > ema_50_4h_aligned[i] and volume_spike[i]:
                signals[i] = 0.20
                position = 1
                highest_high_since_entry = high[i]
            # Short: breakout down AND downtrend AND volume spike
            elif breakout_down and close[i] < ema_50_4h_aligned[i] and volume_spike[i]:
                signals[i] = -0.20
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
                signals[i] = 0.20
        elif position == -1:
            # Update lowest low since entry
            lowest_low_since_entry = min(lowest_low_since_entry, low[i])
            # Exit short: price rises above lowest_low + 2.0 * ATR (trailing stop)
            if close[i] > lowest_low_since_entry + 2.0 * atr_20[i]:
                signals[i] = 0.0
                position = 0
                lowest_low_since_entry = 0.0
            else:
                signals[i] = -0.20
    
    return signals
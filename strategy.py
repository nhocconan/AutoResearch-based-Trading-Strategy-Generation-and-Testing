#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1h Camarilla R3/S3 breakout with 4h EMA34 trend filter and volume confirmation
# Long when price breaks above Camarilla R3 AND price > 4h EMA34 (uptrend) AND volume > 2.0 * 24-period avg volume
# Short when price breaks below Camarilla S3 AND price < 4h EMA34 (downtrend) AND volume > 2.0 * 24-period avg volume
# Exit with ATR-based trailing stop: signal→0 when long and price < highest_high - 2.5 * ATR OR short and price > lowest_low + 2.5 * ATR
# Uses discrete sizing 0.20 to manage drawdown (BTC -77% in 2022 → ~15.4% loss at 0.20 exposure)
# Target: 60-150 total trades over 4 years (15-37/year) for 1h timeframe
# Camarilla pivots provide structured support/resistance levels that work in ranging and trending markets
# 4h EMA34 filters for primary trend direction to avoid counter-trend trades
# High volume threshold (2.0x) ensures strong conviction behind breakouts, reducing false signals
# Works in bull via buying R3 breakouts in uptrend, works in bear via selling S3 breakdowns in downtrend

name = "1h_Camarilla_R3S3_4hEMA34_Volume_v1"
timeframe = "1h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Calculate 1h Camarilla levels (using previous day's OHLC)
    # Camarilla: R4 = close + 1.5*(high-low), R3 = close + 1.1*(high-low), etc.
    # We need previous day's OHLC, so we'll calculate daily OHLC first
    # But since we're on 1h timeframe, we can use rolling window of 24 bars (24h = 1d)
    lookback = 24  # 24 * 1h = 24h = 1 day
    if n < lookback:
        return np.zeros(n)
    
    # Calculate rolling 24-period high, low, close (previous day's values)
    # Shift by 1 to use previous day's data, not current forming day
    prev_high = pd.Series(high).rolling(window=lookback, min_periods=lookback).max().shift(1).values
    prev_low = pd.Series(low).rolling(window=lookback, min_periods=lookback).min().shift(1).values
    prev_close = pd.Series(close).rolling(window=lookback, min_periods=lookback).last().shift(1).values
    
    # Camarilla R3 and S3 levels
    rang = prev_high - prev_low
    camarilla_r3 = prev_close + 1.1 * rang
    camarilla_s3 = prev_close - 1.1 * rang
    
    # Get 4h data ONCE before loop for EMA34 trend filter
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 50:
        return np.zeros(n)
    close_4h = df_4h['close'].values
    
    # Calculate 4h EMA34
    close_4h_series = pd.Series(close_4h)
    ema_34_4h = close_4h_series.ewm(span=34, adjust=False, min_periods=34).mean().values
    
    # Calculate 1h ATR(14) for stoploss
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]  # First period TR is just high-low
    atr_14 = pd.Series(tr).ewm(span=14, adjust=False, min_periods=14).mean().values
    
    # Calculate volume confirmation: volume > 2.0 * 24-period average volume
    avg_volume_24 = pd.Series(volume).rolling(window=24, min_periods=24).mean().values
    volume_spike = volume > (2.0 * avg_volume_24)
    
    # Align HTF indicators to 1h timeframe (wait for completed HTF bar)
    ema_34_4h_aligned = align_htf_to_ltf(prices, df_4h, ema_34_4h)
    
    # Session filter: 08-20 UTC (precompute hours array)
    hours = prices.index.hour  # prices.index is DatetimeIndex
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    highest_high_since_entry = 0.0
    lowest_low_since_entry = 0.0
    
    for i in range(100, n):  # Start after warmup period
        # Skip if any value is NaN
        if (np.isnan(camarilla_r3[i]) or np.isnan(camarilla_s3[i]) or np.isnan(ema_34_4h_aligned[i]) or
            np.isnan(atr_14[i]) or np.isnan(volume_spike[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
                highest_high_since_entry = 0.0
                lowest_low_since_entry = 0.0
            continue
        
        # Session filter: only trade between 08:00 and 20:00 UTC
        hour = hours[i]
        if hour < 8 or hour > 20:
            if position != 0:
                signals[i] = 0.0  # Close position outside session
                position = 0
                highest_high_since_entry = 0.0
                lowest_low_since_entry = 0.0
            else:
                signals[i] = 0.0
            continue
        
        if position == 0:
            # Camarilla breakout signals with trend and volume filters
            breakout_up = close[i] > camarilla_r3[i]   # Price breaks above R3
            breakout_down = close[i] < camarilla_s3[i]  # Price breaks below S3
            
            # Long: breakout up AND uptrend AND volume spike
            if breakout_up and close[i] > ema_34_4h_aligned[i] and volume_spike[i]:
                signals[i] = 0.20
                position = 1
                highest_high_since_entry = high[i]
            # Short: breakout down AND downtrend AND volume spike
            elif breakout_down and close[i] < ema_34_4h_aligned[i] and volume_spike[i]:
                signals[i] = -0.20
                position = -1
                lowest_low_since_entry = low[i]
        elif position == 1:
            # Update highest high since entry
            highest_high_since_entry = max(highest_high_since_entry, high[i])
            # Exit long: price drops below highest_high - 2.5 * ATR (trailing stop)
            if close[i] < highest_high_since_entry - 2.5 * atr_14[i]:
                signals[i] = 0.0
                position = 0
                highest_high_since_entry = 0.0
                lowest_low_since_entry = 0.0
            else:
                signals[i] = 0.20
        elif position == -1:
            # Update lowest low since entry
            lowest_low_since_entry = min(lowest_low_since_entry, low[i])
            # Exit short: price rises above lowest_low + 2.5 * ATR (trailing stop)
            if close[i] > lowest_low_since_entry + 2.5 * atr_14[i]:
                signals[i] = 0.0
                position = 0
                highest_high_since_entry = 0.0
                lowest_low_since_entry = 0.0
            else:
                signals[i] = -0.20
    
    return signals
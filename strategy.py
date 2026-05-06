#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1h Camarilla R3/S3 breakout with 4h EMA50 trend filter and volume confirmation
# Long when price breaks above Camarilla R3 AND price > 4h EMA50 (uptrend) AND volume > 2.0 * 20-period avg volume
# Short when price breaks below Camarilla S3 AND price < 4h EMA50 (downtrend) AND volume > 2.0 * 20-period avg volume
# Exit with ATR-based trailing stop: signal→0 when long and price < highest_high - 2.0 * ATR OR short and price > lowest_low + 2.0 * ATR
# Uses discrete sizing 0.20 to manage drawdown (BTC -77% in 2022 → ~15.4% loss at 0.20 exposure)
# Target: 60-150 total trades over 4 years (15-37/year) for 1h timeframe
# Camarilla levels provide intraday support/resistance, 4h EMA50 filters trend, volume confirmation ensures conviction

name = "1h_Camarilla_R3S3_4hEMA50_Volume_v1"
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
    
    # Get 4h data ONCE before loop for EMA50 trend filter
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 50:
        return np.zeros(n)
    close_4h = df_4h['close'].values
    
    # Calculate 4h EMA50
    close_4h_series = pd.Series(close_4h)
    ema_50_4h = close_4h_series.ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Calculate Camarilla levels for 1h timeframe
    # Based on previous day's OHLC
    daily_high = pd.Series(high).rolling(window=24, min_periods=24).max().shift(24).values  # Previous day high
    daily_low = pd.Series(low).rolling(window=24, min_periods=24).min().shift(24).values     # Previous day low
    daily_close = pd.Series(close).rolling(window=24, min_periods=24).mean().shift(24).values # Previous day close
    
    # Camarilla R3 and S3 levels
    camarilla_range = daily_high - daily_low
    r3 = daily_close + camarilla_range * 1.1 / 4
    s3 = daily_close - camarilla_range * 1.1 / 4
    
    # Calculate 1h ATR(14) for stoploss
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]  # First period TR is just high-low
    atr_14 = pd.Series(tr).ewm(span=14, adjust=False, min_periods=14).mean().values
    
    # Calculate volume confirmation: volume > 2.0 * 20-period average volume
    avg_volume_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (2.0 * avg_volume_20)
    
    # Align HTF indicators to 1h timeframe (wait for completed HTF bar)
    ema_50_4h_aligned = align_htf_to_ltf(prices, df_4h, ema_50_4h)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    highest_high_since_entry = 0.0
    lowest_low_since_entry = 0.0
    
    for i in range(50, n):  # Start after warmup period
        # Skip if any value is NaN
        if (np.isnan(ema_50_4h_aligned[i]) or np.isnan(r3[i]) or np.isnan(s3[i]) or 
            np.isnan(atr_14[i]) or np.isnan(volume_spike[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
                highest_high_since_entry = 0.0
                lowest_low_since_entry = 0.0
            continue
        
        if position == 0:
            # Camarilla breakout signals with trend and volume filters
            breakout_up = close[i] > r3[i-1]  # Use previous bar's level to avoid look-ahead
            breakout_down = close[i] < s3[i-1]  # Use previous bar's level to avoid look-ahead
            
            # Long: breakout up AND uptrend AND volume spike
            if breakout_up and close[i] > ema_50_4h_aligned[i] and volume_spike[i]:
                signals[i] = 0.20
                position = 1
                highest_high_since_entry = close[i]
            # Short: breakout down AND downtrend AND volume spike
            elif breakout_down and close[i] < ema_50_4h_aligned[i] and volume_spike[i]:
                signals[i] = -0.20
                position = -1
                lowest_low_since_entry = close[i]
        elif position == 1:
            # Update highest high since entry
            highest_high_since_entry = max(highest_high_since_entry, close[i])
            # Exit long: price drops below highest_high - 2.0 * ATR (trailing stop)
            if close[i] < highest_high_since_entry - 2.0 * atr_14[i]:
                signals[i] = 0.0
                position = 0
                highest_high_since_entry = 0.0
            else:
                signals[i] = 0.20
        elif position == -1:
            # Update lowest low since entry
            lowest_low_since_entry = min(lowest_low_since_entry, close[i])
            # Exit short: price rises above lowest_low + 2.0 * ATR (trailing stop)
            if close[i] > lowest_low_since_entry + 2.0 * atr_14[i]:
                signals[i] = 0.0
                position = 0
                lowest_low_since_entry = 0.0
            else:
                signals[i] = -0.20
    
    return signals
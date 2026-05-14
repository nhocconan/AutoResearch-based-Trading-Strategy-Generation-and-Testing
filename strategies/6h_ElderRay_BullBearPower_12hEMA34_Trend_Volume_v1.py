#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Elder Ray (Bull/Bear Power) with 12h EMA34 trend filter and volume confirmation
# Bull Power = High - EMA13, Bear Power = EMA13 - Low
# Long when Bull Power > 0 AND Bear Power < 0 (bullish momentum) AND price > 12h EMA34 (uptrend) AND volume > 1.5 * 20-period avg volume
# Short when Bull Power < 0 AND Bear Power > 0 (bearish momentum) AND price < 12h EMA34 (downtrend) AND volume > 1.5 * 20-period avg volume
# Exit with ATR-based trailing stop: signal→0 when long and price < highest_high - 2.0 * ATR OR short and price > lowest_low + 2.0 * ATR
# Uses discrete sizing 0.25 to control drawdown (BTC -77% in 2022 → ~19% loss at 0.25 exposure)
# Target: 50-150 total trades over 4 years (12-37/year) for 6h timeframe
# Elder Ray measures bull/bear power relative to EMA13, 12h EMA34 filters primary trend, volume confirms strength

name = "6h_ElderRay_BullBearPower_12hEMA34_Trend_Volume_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 12h data ONCE before loop for EMA34 trend filter
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 40:
        return np.zeros(n)
    close_12h = df_12h['close'].values
    
    # Calculate 12h EMA34
    close_12h_series = pd.Series(close_12h)
    ema_34_12h = close_12h_series.ewm(span=34, adjust=False, min_periods=34).mean().values
    
    # Get 6h data ONCE before loop for Elder Ray and ATR
    df_6h = get_htf_data(prices, '6h')
    if len(df_6h) < 20:
        return np.zeros(n)
    high_6h = df_6h['high'].values
    low_6h = df_6h['low'].values
    close_6h = df_6h['close'].values
    
    # Calculate 6h EMA13 for Elder Ray
    close_6h_series = pd.Series(close_6h)
    ema_13_6h = close_6h_series.ewm(span=13, adjust=False, min_periods=13).mean().values
    
    # Calculate Elder Ray components
    bull_power = high_6h - ema_13_6h  # Bull Power = High - EMA13
    bear_power = ema_13_6h - low_6h   # Bear Power = EMA13 - Low
    
    # Calculate 6h ATR(10) for stoploss
    tr1 = high_6h - low_6h
    tr2 = np.abs(high_6h - np.roll(close_6h, 1))
    tr3 = np.abs(low_6h - np.roll(close_6h, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]  # First period TR is just high-low
    atr_10 = pd.Series(tr).ewm(span=10, adjust=False, min_periods=10).mean().values
    
    # Calculate volume confirmation: volume > 1.5 * 20-period average volume
    avg_volume_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (1.5 * avg_volume_20)
    
    # Align HTF indicators to 6h timeframe (wait for completed HTF bar)
    ema_34_12h_aligned = align_htf_to_ltf(prices, df_12h, ema_34_12h)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    highest_high_since_entry = 0.0
    lowest_low_since_entry = 0.0
    
    for i in range(40, n):  # Start after warmup period
        # Skip if any value is NaN
        if (np.isnan(ema_34_12h_aligned[i]) or np.isnan(bull_power[i]) or 
            np.isnan(bear_power[i]) or np.isnan(atr_10[i]) or np.isnan(volume_spike[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
                highest_high_since_entry = 0.0
                lowest_low_since_entry = 0.0
            continue
        
        if position == 0:
            # Long: Bull Power > 0 AND Bear Power < 0 (bullish momentum) AND uptrend AND volume spike
            if (bull_power[i] > 0 and bear_power[i] < 0 and 
                close[i] > ema_34_12h_aligned[i] and volume_spike[i]):
                signals[i] = 0.25
                position = 1
                highest_high_since_entry = close[i]
            # Short: Bull Power < 0 AND Bear Power > 0 (bearish momentum) AND downtrend AND volume spike
            elif (bull_power[i] < 0 and bear_power[i] > 0 and 
                  close[i] < ema_34_12h_aligned[i] and volume_spike[i]):
                signals[i] = -0.25
                position = -1
                lowest_low_since_entry = close[i]
        elif position == 1:
            # Update highest high since entry
            highest_high_since_entry = max(highest_high_since_entry, close[i])
            # Exit long: price drops below highest_high - 2.0 * ATR (trailing stop)
            if close[i] < highest_high_since_entry - 2.0 * atr_10[i]:
                signals[i] = 0.0
                position = 0
                highest_high_since_entry = 0.0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Update lowest low since entry
            lowest_low_since_entry = min(lowest_low_since_entry, close[i])
            # Exit short: price rises above lowest_low + 2.0 * ATR (trailing stop)
            if close[i] > lowest_low_since_entry + 2.0 * atr_10[i]:
                signals[i] = 0.0
                position = 0
                lowest_low_since_entry = 0.0
            else:
                signals[i] = -0.25
    
    return signals
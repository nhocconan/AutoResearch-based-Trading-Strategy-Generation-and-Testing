#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Camarilla R1/S1 breakout with 12h EMA50 trend filter and volume confirmation
# Long when price breaks above Camarilla R1 (1.0833*PP - 0.0833*LOW) AND price > 12h EMA50 (uptrend) AND volume > 2.0 * 20-period avg volume
# Short when price breaks below Camarilla S1 (1.0833*HIGH - 0.0833*PP) AND price < 12h EMA50 (downtrend) AND volume > 2.0 * 20-period avg volume
# Exit with ATR-based trailing stop: signal→0 when long and price < highest_high - 2.5 * ATR OR short and price > lowest_low + 2.5 * ATR
# Uses discrete sizing 0.25 to manage drawdown (BTC -77% in 2022 → ~19.25% loss at 0.25 exposure)
# Target: 75-200 total trades over 4 years (19-50/year) for 4h timeframe
# Camarilla provides precise pivot levels, 12h EMA50 filters primary trend, high volume threshold ensures conviction
# Works in bull via breakout continuation, works in bear via mean reversion at extreme levels

name = "4h_Camarilla_R1S1_12hEMA50_Volume_v1"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 12h data ONCE before loop for EMA50 trend filter
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 50:
        return np.zeros(n)
    close_12h = df_12h['close'].values
    
    # Calculate 12h EMA50
    close_12h_series = pd.Series(close_12h)
    ema_50_12h = close_12h_series.ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Calculate 4h ATR(14) for stoploss
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]  # First period TR is just high-low
    atr_14 = pd.Series(tr).ewm(span=14, adjust=False, min_periods=14).mean().values
    
    # Calculate volume confirmation: volume > 2.0 * 20-period average volume
    avg_volume_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (2.0 * avg_volume_20)
    
    # Calculate Camarilla pivot levels (using previous period's OHLC)
    # Pivot Point = (HIGH + LOW + CLOSE) / 3
    # R1 = PP + (HIGH - LOW) * 1.0833 / 2 = 1.0833*HIGH - 0.0833*LOW
    # S1 = PP - (HIGH - LOW) * 1.0833 / 2 = 1.0833*LOW - 0.0833*HIGH
    # Using previous bar's OHLC to avoid look-ahead
    camarilla_r1 = 1.0833 * np.roll(high, 1) - 0.0833 * np.roll(low, 1)
    camarilla_s1 = 1.0833 * np.roll(low, 1) - 0.0833 * np.roll(high, 1)
    
    # Align HTF indicators to 4h timeframe (wait for completed HTF bar)
    ema_50_12h_aligned = align_htf_to_ltf(prices, df_12h, ema_50_12h)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    highest_high_since_entry = 0.0
    lowest_low_since_entry = 0.0
    
    for i in range(50, n):  # Start after warmup period
        # Skip if any value is NaN
        if (np.isnan(ema_50_12h_aligned[i]) or np.isnan(atr_14[i]) or np.isnan(volume_spike[i]) or
            np.isnan(camarilla_r1[i]) or np.isnan(camarilla_s1[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
                highest_high_since_entry = 0.0
                lowest_low_since_entry = 0.0
            continue
        
        if position == 0:
            # Camarilla R1/S1 breakout signals with trend and volume filters
            breakout_up = close[i] > camarilla_r1[i]  # Break above R1
            breakout_down = close[i] < camarilla_s1[i]  # Break below S1
            
            # Long: breakout above R1 AND uptrend AND volume spike
            if breakout_up and close[i] > ema_50_12h_aligned[i] and volume_spike[i]:
                signals[i] = 0.25
                position = 1
                highest_high_since_entry = close[i]
            # Short: breakout below S1 AND downtrend AND volume spike
            elif breakout_down and close[i] < ema_50_12h_aligned[i] and volume_spike[i]:
                signals[i] = -0.25
                position = -1
                lowest_low_since_entry = close[i]
        elif position == 1:
            # Update highest high since entry
            highest_high_since_entry = max(highest_high_since_entry, high[i])
            # Exit long: price drops below highest_high - 2.5 * ATR (trailing stop)
            if close[i] < highest_high_since_entry - 2.5 * atr_14[i]:
                signals[i] = 0.0
                position = 0
                highest_high_since_entry = 0.0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Update lowest low since entry
            lowest_low_since_entry = min(lowest_low_since_entry, low[i])
            # Exit short: price rises above lowest_low + 2.5 * ATR (trailing stop)
            if close[i] > lowest_low_since_entry + 2.5 * atr_14[i]:
                signals[i] = 0.0
                position = 0
                lowest_low_since_entry = 0.0
            else:
                signals[i] = -0.25
    
    return signals
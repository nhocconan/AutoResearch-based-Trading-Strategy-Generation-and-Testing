#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1h Camarilla R3/S3 breakout with 4h EMA50 trend filter and volume confirmation
# Long when price > R3 AND close > 4h EMA50 (uptrend) AND volume > 1.8 * 20-period avg volume
# Short when price < S3 AND close < 4h EMA50 (downtrend) AND volume > 1.8 * 20-period avg volume
# Exit with ATR-based trailing stop: signal→0 when long and price < highest_high - 2.0 * ATR OR short and price > lowest_low + 2.0 * ATR
# Uses discrete sizing 0.20 to control drawdown (BTC -77% in 2022 → ~15% loss at 0.20 exposure)
# Target: 60-150 total trades over 4 years (15-37/year) for 1h timeframe
# Camarilla pivot levels provide intraday support/resistance, 4h EMA50 filters primary trend, volume confirms breakout strength

name = "1h_Camarilla_R3S3_Breakout_4hEMA50_Trend_Volume_v1"
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
    
    # Get 4h data ONCE before loop for EMA50 trend filter
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 50:
        return np.zeros(n)
    close_4h = df_4h['close'].values
    
    # Calculate 4h EMA50
    close_4h_series = pd.Series(close_4h)
    ema_50_4h = close_4h_series.ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Align 4h EMA50 to 1h timeframe (wait for completed 4h bar)
    ema_50_4h_aligned = align_htf_to_ltf(prices, df_4h, ema_50_4h)
    
    # Calculate Camarilla pivot levels for 1h timeframe
    # R3 = close + 1.1 * (high - low) * 1.1/4
    # S3 = close - 1.1 * (high - low) * 1.1/4
    # Using typical Camarilla formula: R4 = close + 1.1*(high-low), R3 = close + 1.1*(high-low)*1.1/4
    # Simplified: R3 = close + 1.1*(high-low)*0.275, S3 = close - 1.1*(high-low)*0.275
    hl_range = high - low
    camarilla_multiplier = 1.1 * 0.275  # 0.3025
    r3 = close + camarilla_multiplier * hl_range
    s3 = close - camarilla_multiplier * hl_range
    
    # Calculate 1h ATR(10) for stoploss
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]  # First period TR is just high-low
    atr_10 = pd.Series(tr).ewm(span=10, adjust=False, min_periods=10).mean().values
    
    # Calculate volume confirmation: volume > 1.8 * 20-period average volume
    avg_volume_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (1.8 * avg_volume_20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    highest_high_since_entry = 0.0
    lowest_low_since_entry = 0.0
    
    for i in range(50, n):  # Start after warmup period
        # Skip if any value is NaN
        if (np.isnan(ema_50_4h_aligned[i]) or np.isnan(r3[i]) or np.isnan(s3[i]) or 
            np.isnan(atr_10[i]) or np.isnan(volume_spike[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
                highest_high_since_entry = 0.0
                lowest_low_since_entry = 0.0
            continue
        
        if position == 0:
            # Long: price > R3 AND uptrend AND volume spike
            if (close[i] > r3[i] and close[i] > ema_50_4h_aligned[i] and volume_spike[i]):
                signals[i] = 0.20
                position = 1
                highest_high_since_entry = close[i]
            # Short: price < S3 AND downtrend AND volume spike
            elif (close[i] < s3[i] and close[i] < ema_50_4h_aligned[i] and volume_spike[i]):
                signals[i] = -0.20
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
                signals[i] = 0.20
        elif position == -1:
            # Update lowest low since entry
            lowest_low_since_entry = min(lowest_low_since_entry, close[i])
            # Exit short: price rises above lowest_low + 2.0 * ATR (trailing stop)
            if close[i] > lowest_low_since_entry + 2.0 * atr_10[i]:
                signals[i] = 0.0
                position = 0
                lowest_low_since_entry = 0.0
            else:
                signals[i] = -0.20
    
    return signals
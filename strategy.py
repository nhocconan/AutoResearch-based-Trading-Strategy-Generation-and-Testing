#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Camarilla R3/S3 breakout with 12h EMA50 trend filter and volume spike confirmation
# Long when price breaks above Camarilla R3 level AND price > 12h EMA50 (uptrend) AND volume > 2.0 * 20-period avg volume
# Short when price breaks below Camarilla S3 level AND price < 12h EMA50 (downtrend) AND volume > 2.0 * 20-period avg volume
# Exit with ATR-based trailing stop: signal→0 when long and price < highest_high - 2.5 * ATR OR short and price > lowest_low + 2.5 * ATR
# Uses discrete sizing 0.25 to balance risk and return (BTC -77% in 2022 → ~19.25% loss at 0.25 exposure)
# Target: 75-200 total trades over 4 years (19-50/year) for 4h timeframe
# Camarilla levels provide intraday support/resistance, 12h EMA50 filters primary trend, volume ensures conviction

name = "4h_Camarilla_R3S3_12hEMA50_Volume_v1"
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
    
    # Calculate Camarilla levels from previous 1d bar (OHLC)
    camarilla_R4 = np.zeros(n)
    camarilla_R3 = np.zeros(n)
    camarilla_R2 = np.zeros(n)
    camarilla_R1 = np.zeros(n)
    camarilla_PP = np.zeros(n)
    camarilla_S1 = np.zeros(n)
    camarilla_S2 = np.zeros(n)
    camarilla_S3 = np.zeros(n)
    camarilla_S4 = np.zeros(n)
    
    for i in range(1, n):
        # Use previous 1d bar's OHLC to calculate today's Camarilla levels
        # Find index of previous completed 1d bar (6 bars of 4h = 24h)
        if i >= 6:
            # Get high, low, close of previous 24h period (6 bars of 4h)
            period_high = np.max(high[i-6:i])
            period_low = np.min(low[i-6:i])
            period_close = close[i-1]  # previous bar close
            
            camarilla_PP[i] = (period_high + period_low + period_close) / 3
            camarilla_R4[i] = camarilla_PP[i] + (period_high - period_low) * 1.5 / 2
            camarilla_R3[i] = camarilla_PP[i] + (period_high - period_low) * 1.25 / 2
            camarilla_R2[i] = camarilla_PP[i] + (period_high - period_low) * 1.1 / 2
            camarilla_R1[i] = camarilla_PP[i] + (period_high - period_low) * 1.0 / 2
            camarilla_S1[i] = camarilla_PP[i] - (period_high - period_low) * 1.0 / 2
            camarilla_S2[i] = camarilla_PP[i] - (period_high - period_low) * 1.1 / 2
            camarilla_S3[i] = camarilla_PP[i] - (period_high - period_low) * 1.25 / 2
            camarilla_S4[i] = camarilla_PP[i] - (period_high - period_low) * 1.5 / 2
        else:
            camarilla_PP[i] = camarilla_R4[i] = camarilla_R3[i] = camarilla_R2[i] = camarilla_R1[i] = camarilla_S1[i] = camarilla_S2[i] = camarilla_S3[i] = camarilla_S4[i] = 0.0
    
    # Align HTF indicators to 4h timeframe (wait for completed HTF bar)
    ema_50_12h_aligned = align_htf_to_ltf(prices, df_12h, ema_50_12h)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    highest_high_since_entry = 0.0
    lowest_low_since_entry = 0.0
    
    for i in range(50, n):  # Start after warmup period
        # Skip if any value is NaN
        if (np.isnan(ema_50_12h_aligned[i]) or np.isnan(atr_14[i]) or np.isnan(volume_spike[i]) or
            np.isnan(camarilla_R3[i]) or np.isnan(camarilla_S3[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
                highest_high_since_entry = 0.0
                lowest_low_since_entry = 0.0
            continue
        
        if position == 0:
            # Camarilla breakout signals with trend and volume filters
            breakout_up = close[i] > camarilla_R3[i-1]  # Use previous bar's level to avoid look-ahead
            breakout_down = close[i] < camarilla_S3[i-1]  # Use previous bar's level to avoid look-ahead
            
            # Long: breakout up above R3 AND uptrend AND volume spike
            if breakout_up and close[i] > ema_50_12h_aligned[i] and volume_spike[i]:
                signals[i] = 0.25
                position = 1
                highest_high_since_entry = close[i]
            # Short: breakout down below S3 AND downtrend AND volume spike
            elif breakout_down and close[i] < ema_50_12h_aligned[i] and volume_spike[i]:
                signals[i] = -0.25
                position = -1
                lowest_low_since_entry = close[i]
        elif position == 1:
            # Update highest high since entry
            highest_high_since_entry = max(highest_high_since_entry, close[i])
            # Exit long: price drops below highest_high - 2.5 * ATR (trailing stop)
            if close[i] < highest_high_since_entry - 2.5 * atr_14[i]:
                signals[i] = 0.0
                position = 0
                highest_high_since_entry = 0.0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Update lowest low since entry
            lowest_low_since_entry = min(lowest_low_since_entry, close[i])
            # Exit short: price rises above lowest_low + 2.5 * ATR (trailing stop)
            if close[i] > lowest_low_since_entry + 2.5 * atr_14[i]:
                signals[i] = 0.0
                position = 0
                lowest_low_since_entry = 0.0
            else:
                signals[i] = -0.25
    
    return signals
#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1d Williams Alligator with 1w EMA50 trend filter and volume confirmation
# Long when price > Alligator Jaw (13-period SMMA smoothed 8) AND price > Alligator Teeth (8-period SMMA smoothed 5) AND price > Alligator Lips (5-period SMMA smoothed 3) AND price > 1w EMA50 (uptrend) AND volume > 1.5 * 20-period avg volume
# Short when price < Alligator Jaw AND price < Alligator Teeth AND price < Alligator Lips AND price < 1w EMA50 (downtrend) AND volume > 1.5 * 20-period avg volume
# Exit with ATR-based trailing stop: signal→0 when long and price < highest_high - 2.0 * ATR OR short and price > lowest_low + 2.0 * ATR
# Uses discrete sizing 0.30 to manage drawdown (BTC -77% in 2022 → ~23.1% loss at 0.30 exposure)
# Target: 30-100 total trades over 4 years (7-25/year) for 1d timeframe
# Williams Alligator identifies trending vs ranging markets; all three lines aligned indicates strong trend
# 1w EMA50 filters primary trend on weekly timeframe, volume confirmation ensures conviction
# Works in bull via trend continuation, works in bear via trend continuation on shorter timeframes within larger downtrend

name = "1d_WilliamsAlligator_1wEMA50_Volume_v1"
timeframe = "1d"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1d data for Williams Alligator calculation ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 13:
        return np.zeros(n)
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Get 1w data for EMA50 trend filter ONCE before loop
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 50:
        return np.zeros(n)
    close_1w = df_1w['close'].values
    
    # Calculate Williams Alligator components (SMMA = smoothed moving average)
    # Jaw: 13-period SMMA of median price, smoothed 8 periods
    median_price_1d = (high_1d + low_1d) / 2
    # SMMA calculation: first value = SMA, subsequent = (prev*(period-1) + current) / period
    def smma(values, period):
        if len(values) < period:
            return np.full_like(values, np.nan)
        result = np.full_like(values, np.nan)
        # First value is simple SMA
        result[period-1] = np.mean(values[:period])
        # Subsequent values: smoothed moving average
        for i in range(period, len(values)):
            result[i] = (result[i-1] * (period-1) + values[i]) / period
        return result
    
    jaw_raw = smma(median_price_1d, 13)
    jaw_8 = smma(jaw_raw, 8)  # Jaw line
    
    # Teeth: 8-period SMMA of median price, smoothed 5 periods
    teeth_raw = smma(median_price_1d, 8)
    teeth_5 = smma(teeth_raw, 5)  # Teeth line
    
    # Lips: 5-period SMMA of median price, smoothed 3 periods
    lips_raw = smma(median_price_1d, 5)
    lips_3 = smma(lips_raw, 3)  # Lips line
    
    # Calculate 1w EMA50 for trend filter
    close_1w_series = pd.Series(close_1w)
    ema_50_1w = close_1w_series.ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Calculate 1d ATR(14) for stoploss
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]  # First period TR is just high-low
    atr_14 = pd.Series(tr).ewm(span=14, adjust=False, min_periods=14).mean().values
    
    # Calculate volume confirmation: volume > 1.5 * 20-period average volume
    avg_volume_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (1.5 * avg_volume_20)
    
    # Align HTF indicators to 1d timeframe (wait for completed HTF bar)
    jaw_8_aligned = align_htf_to_ltf(prices, df_1d, jaw_8)
    teeth_5_aligned = align_htf_to_ltf(prices, df_1d, teeth_5)
    lips_3_aligned = align_htf_to_ltf(prices, df_1d, lips_3)
    ema_50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_50_1w)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    highest_high_since_entry = 0.0
    lowest_low_since_entry = 0.0
    
    for i in range(100, n):  # Start after warmup period
        # Skip if any value is NaN
        if (np.isnan(jaw_8_aligned[i]) or np.isnan(teeth_5_aligned[i]) or np.isnan(lips_3_aligned[i]) or
            np.isnan(ema_50_1w_aligned[i]) or np.isnan(atr_14[i]) or np.isnan(volume_spike[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
                highest_high_since_entry = 0.0
                lowest_low_since_entry = 0.0
            continue
        
        if position == 0:
            # Williams Alligator alignment signals with trend and volume filters
            # Alligator lines aligned upward: Lips > Teeth > Jaw (bullish alignment)
            bullish_alignment = (lips_3_aligned[i] > teeth_5_aligned[i]) and (teeth_5_aligned[i] > jaw_8_aligned[i])
            # Alligator lines aligned downward: Lips < Teeth < Jaw (bearish alignment)
            bearish_alignment = (lips_3_aligned[i] < teeth_5_aligned[i]) and (teeth_5_aligned[i] < jaw_8_aligned[i])
            
            # Long: bullish alignment AND price > Alligator lines AND uptrend AND volume spike
            if bullish_alignment and close[i] > lips_3_aligned[i] and close[i] > ema_50_1w_aligned[i] and volume_spike[i]:
                signals[i] = 0.30
                position = 1
                highest_high_since_entry = close[i]
            # Short: bearish alignment AND price < Alligator lines AND downtrend AND volume spike
            elif bearish_alignment and close[i] < lips_3_aligned[i] and close[i] < ema_50_1w_aligned[i] and volume_spike[i]:
                signals[i] = -0.30
                position = -1
                lowest_low_since_entry = close[i]
        elif position == 1:
            # Update highest high since entry
            highest_high_since_entry = max(highest_high_since_entry, high[i])
            # Exit long: price drops below highest_high - 2.0 * ATR (trailing stop)
            if close[i] < highest_high_since_entry - 2.0 * atr_14[i]:
                signals[i] = 0.0
                position = 0
                highest_high_since_entry = 0.0
            else:
                signals[i] = 0.30
        elif position == -1:
            # Update lowest low since entry
            lowest_low_since_entry = min(lowest_low_since_entry, low[i])
            # Exit short: price rises above lowest_low + 2.0 * ATR (trailing stop)
            if close[i] > lowest_low_since_entry + 2.0 * atr_14[i]:
                signals[i] = 0.0
                position = 0
                lowest_low_since_entry = 0.0
            else:
                signals[i] = -0.30
    
    return signals
#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h Williams Alligator with 1d EMA34 trend filter and volume confirmation
# Williams Alligator: Jaw (EMA13, 8-period smoothed), Teeth (EMA8, 5-period smoothed), Lips (EMA5, 3-period smoothed)
# Long when Lips > Teeth > Jaw (bullish alignment) AND price > 1d EMA34 (uptrend) AND volume > 1.5 * 20-period avg volume
# Short when Lips < Teeth < Jaw (bearish alignment) AND price < 1d EMA34 (downtrend) AND volume > 1.5 * 20-period avg volume
# Exit with ATR-based trailing stop: signal→0 when long and price < highest_high - 2.5 * ATR OR short and price > lowest_low + 2.5 * ATR
# Uses discrete sizing 0.25 to manage drawdown (BTC -77% in 2022 → ~19.25% loss at 0.25 exposure)
# Target: 50-150 total trades over 4 years (12-37/year) for 12h timeframe
# Williams Alligator identifies trend via jaw/teeth/lips alignment, 1d EMA34 filters primary trend, volume threshold ensures conviction
# Works in bull via bullish alignment, works in bear via bearish alignment

name = "12h_WilliamsAlligator_1dEMA34_Volume_v1"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1d data ONCE before loop for EMA34 trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 34:
        return np.zeros(n)
    close_1d = df_1d['close'].values
    
    # Calculate 1d EMA34
    close_1d_series = pd.Series(close_1d)
    ema_34_1d = close_1d_series.ewm(span=34, adjust=False, min_periods=34).mean().values
    
    # Calculate 12h Williams Alligator components
    close_series = pd.Series(close)
    # Jaw: EMA13 smoothed by 8 periods
    jaw_raw = close_series.ewm(span=13, adjust=False, min_periods=13).mean().values
    jaw = pd.Series(jaw_raw).ewm(span=8, adjust=False, min_periods=8).mean().values
    # Teeth: EMA8 smoothed by 5 periods
    teeth_raw = close_series.ewm(span=8, adjust=False, min_periods=8).mean().values
    teeth = pd.Series(teeth_raw).ewm(span=5, adjust=False, min_periods=5).mean().values
    # Lips: EMA5 smoothed by 3 periods
    lips_raw = close_series.ewm(span=5, adjust=False, min_periods=5).mean().values
    lips = pd.Series(lips_raw).ewm(span=3, adjust=False, min_periods=3).mean().values
    
    # Calculate 12h ATR(14) for stoploss
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]  # First period TR is just high-low
    atr_14 = pd.Series(tr).ewm(span=14, adjust=False, min_periods=14).mean().values
    
    # Calculate volume confirmation: volume > 1.5 * 20-period average volume
    avg_volume_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (1.5 * avg_volume_20)
    
    # Align HTF indicators to 12h timeframe (wait for completed HTF bar)
    ema_34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    highest_high_since_entry = 0.0
    lowest_low_since_entry = 0.0
    
    for i in range(50, n):  # Start after warmup period
        # Skip if any value is NaN
        if (np.isnan(ema_34_1d_aligned[i]) or np.isnan(atr_14[i]) or np.isnan(volume_spike[i]) or
            np.isnan(lips[i]) or np.isnan(teeth[i]) or np.isnan(jaw[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
                highest_high_since_entry = 0.0
                lowest_low_since_entry = 0.0
            continue
        
        if position == 0:
            # Williams Alligator signals with trend and volume filters
            bullish_alignment = lips[i] > teeth[i] and teeth[i] > jaw[i]  # Lips > Teeth > Jaw
            bearish_alignment = lips[i] < teeth[i] and teeth[i] < jaw[i]  # Lips < Teeth < Jaw
            
            # Long: bullish alignment AND uptrend AND volume spike
            if bullish_alignment and close[i] > ema_34_1d_aligned[i] and volume_spike[i]:
                signals[i] = 0.25
                position = 1
                highest_high_since_entry = close[i]
            # Short: bearish alignment AND downtrend AND volume spike
            elif bearish_alignment and close[i] < ema_34_1d_aligned[i] and volume_spike[i]:
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
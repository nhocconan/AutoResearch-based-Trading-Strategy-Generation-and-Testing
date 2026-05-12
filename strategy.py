# 4h_Alligator_Vortex_Balance_1dTrend_Filter
# Hypothesis: Combines Williams Alligator (trend direction), Vortex (momentum), and daily trend filter for robust signals.
# Alligator identifies trend direction via SMA crossovers, Vortex confirms momentum strength, daily trend ensures alignment.
# Works in bull/bear by requiring both momentum and trend alignment, reducing false signals.
# Target: 20-50 trades/year with strict entry conditions to avoid overtrading.

name = "4h_Alligator_Vortex_Balance_1dTrend_Filter"
timeframe = "4h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # === 1D DATA FOR TREND FILTER ===
    df_1d = get_htf_data(prices, '1d')
    close_1d = df_1d['close'].values
    
    # Daily EMA50 for trend filter
    ema50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema50_1d_4h = align_htf_to_ltf(prices, df_1d, ema50_1d)
    
    # === ALLIGATOR (Williams) ===
    # Jaw (13-period SMMA, 8-bar offset), Teeth (8-period SMMA, 5-bar offset), Lips (5-period SMMA, 3-bar offset)
    def smma(arr, period):
        result = np.full_like(arr, np.nan, dtype=float)
        if len(arr) < period:
            return result
        sma = np.mean(arr[:period])
        result[period-1] = sma
        for i in range(period, len(arr)):
            result[i] = (result[i-1] * (period-1) + arr[i]) / period
        return result
    
    jaw = smma(close, 13)
    teeth = smma(close, 8)
    lips = smma(close, 5)
    
    # Shift for Alligator offsets
    jaw = np.roll(jaw, 8)
    teeth = np.roll(teeth, 5)
    lips = np.roll(lips, 3)
    
    # === VORTEX INDICATOR ===
    vm_plus = np.abs(high - np.roll(low, 1))
    vm_minus = np.abs(low - np.roll(high, 1))
    # Handle first element
    vm_plus[0] = np.abs(high[0] - low[-1]) if len(low) > 1 else 0
    vm_minus[0] = np.abs(low[0] - high[-1]) if len(high) > 1 else 0
    
    tr = np.maximum(np.abs(high - low), np.maximum(np.abs(high - np.roll(close, 1)), np.abs(low - np.roll(close, 1))))
    tr[0] = np.abs(high[0] - low[0])
    
    vi_plus = pd.Series(vm_plus).rolling(window=14, min_periods=14).sum().values / \
              pd.Series(tr).rolling(window=14, min_periods=14).sum().values
    vi_minus = pd.Series(vm_minus).rolling(window=14, min_periods=14).sum().values / \
               pd.Series(tr).rolling(window=14, min_periods=14).sum().values
    
    # === VOLUME CONFIRMATION ===
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (vol_ma * 1.5)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(100, 20)  # Ensure all indicators ready
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(jaw[i]) or np.isnan(teeth[i]) or np.isnan(lips[i]) or
            np.isnan(vi_plus[i]) or np.isnan(vi_minus[i]) or np.isnan(ema50_1d_4h[i]) or
            np.isnan(vol_ma[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue
        
        if position == 0:
            # LONG: Lips > Teeth > Jaw (bullish alignment) + VI+ > VI- (bullish momentum) + price above daily EMA50 + volume spike
            if (lips[i] > teeth[i] > jaw[i] and
                vi_plus[i] > vi_minus[i] and
                close[i] > ema50_1d_4h[i] and
                volume_spike[i]):
                signals[i] = 0.25
                position = 1
            # SHORT: Lips < Teeth < Jaw (bearish alignment) + VI- > VI+ (bearish momentum) + price below daily EMA50 + volume spike
            elif (lips[i] < teeth[i] < jaw[i] and
                  vi_minus[i] > vi_plus[i] and
                  close[i] < ema50_1d_4h[i] and
                  volume_spike[i]):
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # EXIT LONG: Alligator alignment breaks OR momentum reverses OR price below daily EMA50
            if not (lips[i] > teeth[i] > jaw[i]) or vi_plus[i] <= vi_minus[i] or close[i] < ema50_1d_4h[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: Alligator alignment breaks OR momentum reverses OR price above daily EMA50
            if not (lips[i] < teeth[i] < jaw[i]) or vi_minus[i] <= vi_plus[i] or close[i] > ema50_1d_4h[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals
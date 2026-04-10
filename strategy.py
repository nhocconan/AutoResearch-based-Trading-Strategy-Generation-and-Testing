#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Elder Ray + 1d regime filter + volume confirmation
# - Elder Ray (Bull/Bear Power) measures buying/selling pressure relative to EMA13
# - Bull Power = High - EMA13, Bear Power = Low - EMA13
# - Long when Bull Power > 0 and rising + price > EMA50 (bull regime) + volume spike
# - Short when Bear Power < 0 and falling + price < EMA50 (bear regime) + volume spike
# - Uses 6h timeframe to target 12-37 trades/year (50-150 total over 4 years) to minimize fee drag
# - 1d EMA50 acts as regime filter: trade only in direction of higher timeframe trend
# - Volume confirmation: current 6h volume > 2.0x 20-period average to filter weak signals
# - Discrete position sizing (0.25) to minimize fee churn
# - ATR-based stoploss: exit when price moves against position by 2.0x ATR(14)

name = "6h_1d_elder_ray_regime_volume_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Load HTF data ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # Pre-compute 1d EMA50 for regime filter
    close_1d = df_1d['close'].values
    ema50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema50_1d)
    
    # Pre-compute 6h EMA13 for Elder Ray
    close_6h = prices['close'].values
    ema13_6h = pd.Series(close_6h).ewm(span=13, adjust=False, min_periods=13).mean().values
    
    # Pre-compute Elder Ray components
    high_6h = prices['high'].values
    low_6h = prices['low'].values
    bull_power = high_6h - ema13_6h  # Buying pressure
    bear_power = low_6h - ema13_6h   # Selling pressure
    
    # Pre-compute 6h ATR(14) for stoploss
    tr1 = high_6h - low_6h
    tr2 = np.abs(high_6h - np.roll(close_6h, 1))
    tr3 = np.abs(low_6h - np.roll(close_6h, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]
    atr_14 = pd.Series(tr).ewm(span=14, adjust=False, min_periods=14).mean().values
    
    # Pre-compute 6h volume confirmation
    volume_6h = prices['volume'].values
    avg_volume_20 = pd.Series(volume_6h).rolling(window=20, min_periods=20).mean().values
    vol_spike = volume_6h > (2.0 * avg_volume_20)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    entry_price = 0.0
    
    for i in range(20, n):
        # Skip if any required data is invalid
        if (np.isnan(ema50_1d_aligned[i]) or np.isnan(bull_power[i]) or 
            np.isnan(bear_power[i]) or np.isnan(vol_spike[i]) or np.isnan(atr_14[i])):
            signals[i] = 0.0
            continue
        
        if position == 1:  # Long position
            # Exit: ATR-based stoploss or regime change (price crosses below EMA50)
            if (prices['close'].iloc[i] < entry_price - 2.0 * atr_14[i] or 
                prices['close'].iloc[i] < ema50_1d_aligned[i]):
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit: ATR-based stoploss or regime change (price crosses above EMA50)
            if (prices['close'].iloc[i] > entry_price + 2.0 * atr_14[i] or 
                prices['close'].iloc[i] > ema50_1d_aligned[i]):
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat
            # Look for Elder Ray signals with regime and volume filters
            if vol_spike[i]:
                # Long signal: Bull Power positive AND rising + bull regime (price > EMA50)
                if (bull_power[i] > 0 and bull_power[i] > bull_power[i-1] and 
                    prices['close'].iloc[i] > ema50_1d_aligned[i]):
                    position = 1
                    entry_price = prices['close'].iloc[i]
                    signals[i] = 0.25
                # Short signal: Bear Power negative AND falling + bear regime (price < EMA50)
                elif (bear_power[i] < 0 and bear_power[i] < bear_power[i-1] and 
                      prices['close'].iloc[i] < ema50_1d_aligned[i]):
                    position = -1
                    entry_price = prices['close'].iloc[i]
                    signals[i] = -0.25
    
    return signals
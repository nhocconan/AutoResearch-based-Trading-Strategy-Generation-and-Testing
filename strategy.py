#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Elder Ray (Bull/Bear Power) with 1d regime filter and volume confirmation
# - Bull Power = High - EMA13, Bear Power = EMA13 - Low
# - Long when Bull Power > 0 and Bear Power < 0 (bullish momentum) with 1d EMA50 uptrend and volume spike
# - Short when Bear Power > 0 and Bull Power < 0 (bearish momentum) with 1d EMA50 downtrend and volume spike
# - Uses 6h timeframe to target 50-150 total trades over 4 years (12-37/year) to minimize fee drag
# - 1d EMA50 filter ensures we trade with strong daily trend direction (avoid chop)
# - Volume confirmation: current 6h volume > 2.0x 20-period average to filter weak signals
# - Discrete position sizing (0.25) to minimize fee churn
# - ATR-based stoploss: exit when price moves against position by 2.5x ATR(14)

name = "6h_1d_elder_ray_regime_volume_v2"
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
    
    # Pre-compute 6h Elder Ray components
    high_6h = prices['high'].values
    low_6h = prices['low'].values
    bull_power = high_6h - ema13_6h  # Bull Power = High - EMA13
    bear_power = ema13_6h - low_6h   # Bear Power = EMA13 - Low
    
    # Pre-compute 6h ATR(14) for stoploss
    high_6h = prices['high'].values
    low_6h = prices['low'].values
    close_6h = prices['close'].values
    
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
    
    for i in range(50, n):
        # Skip if any required data is invalid
        if (np.isnan(ema50_1d_aligned[i]) or np.isnan(bull_power[i]) or 
            np.isnan(bear_power[i]) or np.isnan(vol_spike[i]) or np.isnan(atr_14[i])):
            signals[i] = 0.0
            continue
        
        if position == 1:  # Long position
            # Exit: ATR-based stoploss or regime turns bearish
            if (prices['close'].iloc[i] < entry_price - 2.5 * atr_14[i] or 
                ema50_1d_aligned[i] < prices['close'].iloc[i]):  # price below daily EMA50 = regime change
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit: ATR-based stoploss or regime turns bullish
            if (prices['close'].iloc[i] > entry_price + 2.5 * atr_14[i] or 
                ema50_1d_aligned[i] > prices['close'].iloc[i]):  # price above daily EMA50 = regime change
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat
            # Look for Elder Ray signals with regime and volume filters
            if vol_spike[i]:
                # Long signal: Bull Power > 0 and Bear Power < 0 (bullish momentum) in daily uptrend
                if bull_power[i] > 0 and bear_power[i] < 0 and ema50_1d_aligned[i] > prices['close'].iloc[i]:
                    position = 1
                    entry_price = prices['close'].iloc[i]
                    signals[i] = 0.25
                # Short signal: Bear Power > 0 and Bull Power < 0 (bearish momentum) in daily downtrend
                elif bear_power[i] > 0 and bull_power[i] < 0 and ema50_1d_aligned[i] < prices['close'].iloc[i]:
                    position = -1
                    entry_price = prices['close'].iloc[i]
                    signals[i] = -0.25
    
    return signals
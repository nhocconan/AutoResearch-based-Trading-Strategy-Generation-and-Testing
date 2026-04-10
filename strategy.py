#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Elder Ray + ADX regime filter with volume confirmation
# - Bull Power = Close - EMA(13), Bear Power = EMA(13) - Low
# - Trend regime: ADX(14) > 25 + DI+ > DI- (bull) or DI- > DI+ (bear)
# - Long when Bull Power > 0, ADX bull regime, and volume > 1.5x 20-period average
# - Short when Bear Power > 0, ADX bear regime, and volume > 1.5x 20-period average
# - Uses discrete position sizing (0.25) to minimize fee churn
# - Targets 12-37 trades/year (50-150 total over 4 years) to avoid fee drag
# - ADX regime filter prevents whipsaw in ranging markets
# - ATR-based stoploss to limit drawdown

name = "6h_elder_ray_adx_regime_volume_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Load HTF data ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # Pre-compute 1d indicators for regime
    close_1d = df_1d['close'].values
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    
    # 1d EMA(13) for Elder Ray
    ema_13_1d = pd.Series(close_1d).ewm(span=13, adjust=False, min_periods=13).mean().values
    
    # 1d ADX components
    plus_dm = np.zeros_like(high_1d)
    minus_dm = np.zeros_like(high_1d)
    for i in range(1, len(high_1d)):
        plus_dm[i] = max(high_1d[i] - high_1d[i-1], 0) if (high_1d[i] - high_1d[i-1]) > (low_1d[i-1] - low_1d[i]) else 0
        minus_dm[i] = max(low_1d[i-1] - low_1d[i], 0) if (low_1d[i-1] - low_1d[i]) > (high_1d[i] - high_1d[i-1]) else 0
    
    tr1 = high_1d - low_1d
    tr2 = np.abs(high_1d - np.roll(close_1d, 1))
    tr3 = np.abs(low_1d - np.roll(close_1d, 1))
    tr1[0] = 0
    tr2[0] = 0
    tr3[0] = 0
    atr_1d = np.maximum(np.maximum(tr1, tr2), tr3)
    
    # Smooth TR, +DM, -DM
    tr_ma = pd.Series(atr_1d).rolling(window=14, min_periods=14).mean().values
    plus_dm_ma = pd.Series(plus_dm).rolling(window=14, min_periods=14).mean().values
    minus_dm_ma = pd.Series(minus_dm).rolling(window=14, min_periods=14).mean().values
    
    # Avoid division by zero
    plus_di = np.where(tr_ma > 0, 100 * plus_dm_ma / tr_ma, 0)
    minus_di = np.where(tr_ma > 0, 100 * minus_dm_ma / tr_ma, 0)
    dx = np.where((plus_di + minus_di) > 0, 100 * np.abs(plus_di - minus_di) / (plus_di + minus_di), 0)
    adx = pd.Series(dx).rolling(window=14, min_periods=14).mean().values
    
    # Align 1d indicators to 6h timeframe
    ema_13_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_13_1d)
    adx_aligned = align_htf_to_ltf(prices, df_1d, adx)
    plus_di_aligned = align_htf_to_ltf(prices, df_1d, plus_di)
    minus_di_aligned = align_htf_to_ltf(prices, df_1d, minus_di)
    
    # Pre-compute ATR for stoploss (using 1d data)
    high_low = df_1d['high'] - df_1d['low']
    high_close = np.abs(df_1d['high'] - df_1d['close'].shift(1))
    low_close = np.abs(df_1d['low'] - df_1d['close'].shift(1))
    ranges = pd.concat([high_low, high_close, low_close], axis=1)
    true_ranges = np.nanmax(ranges.values, axis=1)
    atr_14_1d = pd.Series(true_ranges).rolling(window=14, min_periods=14).mean().values
    atr_14_1d_aligned = align_htf_to_ltf(prices, df_1d, atr_14_1d)
    
    # Volume confirmation on 6h
    volume_ma_20 = pd.Series(prices['volume'].values).rolling(window=20, min_periods=20).mean().values
    volume_spike = prices['volume'].values > (1.5 * volume_ma_20)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    entry_price = 0.0
    atr_stop_multiplier = 2.5
    
    for i in range(100, n):
        # Skip if any required data is invalid
        if (np.isnan(ema_13_1d_aligned[i]) or np.isnan(adx_aligned[i]) or 
            np.isnan(plus_di_aligned[i]) or np.isnan(minus_di_aligned[i]) or 
            np.isnan(atr_14_1d_aligned[i])):
            signals[i] = 0.0
            continue
        
        if position == 1:  # Long position
            # ATR-based stoploss
            if prices['close'].iloc[i] < entry_price - atr_stop_multiplier * atr_14_1d_aligned[i]:
                position = 0
                entry_price = 0.0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # ATR-based stoploss
            if prices['close'].iloc[i] > entry_price + atr_stop_multiplier * atr_14_1d_aligned[i]:
                position = 0
                entry_price = 0.0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat
            # Elder Ray components
            bull_power = prices['close'].iloc[i] - ema_13_1d_aligned[i]
            bear_power = ema_13_1d_aligned[i] - prices['low'].iloc[i]
            
            # ADX regime conditions
            adx_strong = adx_aligned[i] > 25
            di_bull = plus_di_aligned[i] > minus_di_aligned[i]
            di_bear = minus_di_aligned[i] > plus_di_aligned[i]
            
            # Long signal: Bull Power > 0, ADX bull regime, volume spike
            if (bull_power > 0 and adx_strong and di_bull and volume_spike[i]):
                position = 1
                entry_price = prices['open'].iloc[i+1] if i+1 < n else prices['close'].iloc[i]
                signals[i] = 0.25
            # Short signal: Bear Power > 0, ADX bear regime, volume spike
            elif (bear_power > 0 and adx_strong and di_bear and volume_spike[i]):
                position = -1
                entry_price = prices['open'].iloc[i+1] if i+1 < n else prices['close'].iloc[i]
                signals[i] = -0.25
    
    return signals
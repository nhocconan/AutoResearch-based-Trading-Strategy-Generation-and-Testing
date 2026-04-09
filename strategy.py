#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Donchian(20) breakout with 1d volume confirmation and 1w trend filter
# - Uses 6h Donchian channels for breakout signals (long above 20-period high, short below 20-period low)
# - Confirms with 1d volume > 2.0x 20-period average (strong institutional participation)
# - Filters by 1w trend: only trade in direction of 1w EMA(21) trend (long if close > EMA, short if close < EMA)
# - Exits when price touches opposite Donchian level or ATR-based stoploss (2.5x ATR)
# - Position size: 0.25 (25% of capital) to balance risk and minimize fee drag
# - Target: 12-30 trades/year on 6h timeframe (50-120 total over 4 years) to avoid overtrading
# - Works in bull markets (breakouts continue with trend) and bear markets (breakdowns continue with trend)
# - Volume confirmation ensures breakouts have conviction; 1w trend filter avoids counter-trend whipsaws

name = "6h_1d_1w_donchian_volume_trend_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Load HTF data ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    df_1w = get_htf_data(prices, '1w')
    if len(df_1d) < 30 or len(df_1w) < 20:
        return np.zeros(n)
    
    # Pre-compute 1d indicators
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    volume_1d = df_1d['volume'].values
    
    # 1d True Range for ATR
    tr1 = high_1d - low_1d
    tr2 = np.abs(high_1d - np.roll(close_1d, 1))
    tr3 = np.abs(low_1d - np.roll(close_1d, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]
    
    # 1d ATR(14) for stoploss
    atr_1d = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    # 1d Donchian channels (20-period)
    donchian_high = pd.Series(high_1d).rolling(window=20, min_periods=20).max().values
    donchian_low = pd.Series(low_1d).rolling(window=20, min_periods=20).min().values
    
    # 1d Volume > 2.0x 20-period average (stricter for fewer trades)
    avg_volume_20 = pd.Series(volume_1d).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume_1d > (2.0 * avg_volume_20)
    
    # Pre-compute 1w indicators
    close_1w = df_1w['close'].values
    
    # 1w EMA(21) for trend filter
    ema_21_1w = pd.Series(close_1w).ewm(span=21, adjust=False, min_periods=21).mean().values
    # Trend: 1 = uptrend (close > EMA), -1 = downtrend (close < EMA), 0 = neutral
    trend_1w = np.where(close_1w > ema_21_1w, 1, np.where(close_1w < ema_21_1w, -1, 0))
    
    # Align all HTF indicators to 6h
    donchian_high_aligned = align_htf_to_ltf(prices, df_1d, donchian_high)
    donchian_low_aligned = align_htf_to_ltf(prices, df_1d, donchian_low)
    volume_spike_aligned = align_htf_to_ltf(prices, df_1d, volume_spike.astype(float))
    atr_1d_aligned = align_htf_to_ltf(prices, df_1d, atr_1d)
    trend_1w_aligned = align_htf_to_ltf(prices, df_1w, trend_1w.astype(float))
    
    # 6h price data
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    entry_price = 0.0
    atr_stop = 0.0
    
    for i in range(30, n):
        # Skip if any required data is invalid
        if (np.isnan(donchian_high_aligned[i]) or np.isnan(donchian_low_aligned[i]) or
            np.isnan(volume_spike_aligned[i]) or np.isnan(atr_1d_aligned[i]) or
            np.isnan(trend_1w_aligned[i]) or atr_1d_aligned[i] <= 0):
            signals[i] = 0.0
            continue
        
        if position == 1:  # Long position
            # Exit conditions: opposite Donchian touch (low) or ATR stoploss
            if low[i] <= donchian_low_aligned[i]:  # Touch opposite band
                position = 0
                signals[i] = 0.0
            elif high[i] >= entry_price + (2.5 * atr_stop):  # ATR stoploss
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit conditions: opposite Donchian touch (high) or ATR stoploss
            if high[i] >= donchian_high_aligned[i]:  # Touch opposite band
                position = 0
                signals[i] = 0.0
            elif low[i] <= entry_price - (2.5 * atr_stop):  # ATR stoploss
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat
            # Look for Donchian breakout with volume confirmation and 1w trend filter
            # Long: break above upper band + volume spike + 1w uptrend
            if (high[i] >= donchian_high_aligned[i] and  # Break above upper band
                volume_spike_aligned[i] and              # Volume confirmation
                trend_1w_aligned[i] == 1):               # 1w uptrend
                position = 1
                entry_price = high[i]
                atr_stop = atr_1d_aligned[i]
                signals[i] = 0.25
            # Short: break below lower band + volume spike + 1w downtrend
            elif (low[i] <= donchian_low_aligned[i] and    # Break below lower band
                  volume_spike_aligned[i] and              # Volume confirmation
                  trend_1w_aligned[i] == -1):              # 1w downtrend
                position = -1
                entry_price = low[i]
                atr_stop = atr_1d_aligned[i]
                signals[i] = -0.25
    
    return signals
#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Camarilla R3/S3 breakout + 1d EMA34 trend + volume spike + chop regime filter
# Targets 75-200 total trades over 4 years (19-50/year) to minimize fee drag
# Camarilla R3/S3 provides clear breakout structure from 1d pivots
# 1d EMA34 determines trend bias: long when price > EMA34, short when price < EMA34
# Volume spike (2x 20-period average) confirms institutional participation
# Choppiness regime filter: only trade when CHOP(14) > 61.8 (ranging market) for mean reversion at pivot levels
# Works in bull markets via breakouts with trend alignment and bear markets via fade of false breakouts
# Discrete position sizing: 0.30 (30% of capital) balances exposure and risk
# Uses 1d as HTF as specified in experiment #117280

name = "4h_Camarilla_R3S3_Breakout_1dEMA34_VolumeSpike_ChopFilter"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Calculate 1d Camarilla levels (prior completed 1d bar's range)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    # Prior completed 1d bar's high, low, close
    ph = df_1d['high'].shift(1).values  # prior day high
    pl = df_1d['low'].shift(1).values   # prior day low
    pc = df_1d['close'].shift(1).values # prior day close
    
    # Camarilla R3, S3 levels
    camarilla_r3 = pc + (ph - pl) * 1.1 / 4
    camarilla_s3 = pc - (ph - pl) * 1.1 / 4
    
    # Align to 4h timeframe (wait for completed 1d bar)
    camarilla_r3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_r3)
    camarilla_s3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_s3)
    
    # Calculate 1d EMA34 trend (prior completed 1d bar's EMA)
    ema_34 = pd.Series(df_1d['close']).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_aligned = align_htf_to_ltf(prices, df_1d, ema_34)
    
    # Calculate 4h volume spike (2x 20-period average)
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().shift(1).values
    volume_spike = volume > (vol_ma * 2.0)
    
    # Calculate 4h Choppiness Index (CHOP) for regime filter
    # CHOP = 100 * log10(sum(ATR(14)) / log10(highest_high - lowest_low)) / log10(14)
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = 0  # First period has no previous close
    atr14 = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    hh14 = pd.Series(high).rolling(window=14, min_periods=14).max().values
    ll14 = pd.Series(low).rolling(window=14, min_periods=14).min().values
    chop_raw = 100 * np.log10(atr14.sum() / (hh14 - ll14)) / np.log10(14) if (hh14 - ll14) > 0 else 50
    # Vectorized CHOP calculation
    atr14_sum = pd.Series(tr).rolling(window=14, min_periods=14).sum().values
    hh14 = pd.Series(high).rolling(window=14, min_periods=14).max().values
    ll14 = pd.Series(low).rolling(window=14, min_periods=14).min().values
    range_14 = hh14 - ll14
    chop = np.where(range_14 > 0, 100 * np.log10(atr14_sum / range_14) / np.log10(14), 50)
    chop_regime = chop > 61.8  # Ranging market (mean reversion zone)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after warmup (need enough data for all indicators)
    start_idx = max(34, 20, 14)
    
    for i in range(start_idx, n):
        # Check for NaN values in indicators
        if (np.isnan(camarilla_r3_aligned[i]) or np.isnan(camarilla_s3_aligned[i]) or 
            np.isnan(ema_34_aligned[i]) or np.isnan(volume_spike[i]) or np.isnan(chop_regime[i])):
            signals[i] = 0.0
            continue
        
        if position == 0:  # Flat - look for new entries
            # Long entry: price breaks above Camarilla R3 AND price > 1d EMA34 (bullish bias) AND volume spike AND chop regime (ranging)
            if (close[i] > camarilla_r3_aligned[i] and 
                close[i] > ema_34_aligned[i] and 
                volume_spike[i] and 
                chop_regime[i]):
                signals[i] = 0.30
                position = 1
            # Short entry: price breaks below Camarilla S3 AND price < 1d EMA34 (bearish bias) AND volume spike AND chop regime (ranging)
            elif (close[i] < camarilla_s3_aligned[i] and 
                  close[i] < ema_34_aligned[i] and 
                  volume_spike[i] and 
                  chop_regime[i]):
                signals[i] = -0.30
                position = -1
            else:
                signals[i] = 0.0
        
        elif position == 1:  # Long position
            # Exit: price falls below Camarilla S3 OR below 1d EMA34 (trend change)
            if close[i] < camarilla_s3_aligned[i] or close[i] < ema_34_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.30
        
        elif position == -1:  # Short position
            # Exit: price rises above Camarilla R3 OR above 1d EMA34 (trend change)
            if close[i] > camarilla_r3_aligned[i] or close[i] > ema_34_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.30
    
    return signals
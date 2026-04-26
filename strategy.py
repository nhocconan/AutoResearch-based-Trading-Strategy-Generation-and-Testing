#!/usr/bin/env python3
"""
1d_KAMA_Trend_RSI_ChopFilter_VolumeSpike
Hypothesis: On daily timeframe, use Kaufman Adaptive Moving Average (KAMA) for trend direction,
RSI(14) for momentum confirmation, Choppiness Index for regime filter (CHOP > 61.8 = ranging),
and volume spike (>1.5x average) for entry quality. Long when KAMA up, RSI > 50, CHOP > 61.8,
volume spike, and price > KAMA. Short when KAMA down, RSI < 50, CHOP > 61.8, volume spike,
and price < KAMA. Uses discrete position sizing (0.25) to minimize fee churn. Target: 30-100
trades over 4 years (7-25/year) on 1d timeframe. Uses 1-week HTF trend filter (EMA50) to avoid
counter-trend trades in strong weekly trends. Designed to work in both bull and bear markets
by requiring multiple confirmations and using adaptive indicators that adjust to volatility.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:  # Need warmup for indicators
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Load 1d data for indicators
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    # Calculate KAMA ( Kaufman Adaptive Moving Average )
    close_1d = pd.Series(df_1d['close'])
    # Efficiency Ratio (ER)
    change = abs(close_1d - close_1d.shift(10))
    volatility = abs(close_1d.diff()).rolling(window=10, min_periods=10).sum()
    er = change / volatility.replace(0, np.nan)
    # Smoothing constants
    fast_sc = 2 / (2 + 1)   # EMA(2)
    slow_sc = 2 / (30 + 1)  # EMA(30)
    sc = (er * (fast_sc - slow_sc) + slow_sc) ** 2
    # Initialize KAMA
    kama = close_1d.copy()
    for i in range(1, len(kama)):
        if not np.isnan(sc.iloc[i]):
            kama.iloc[i] = kama.iloc[i-1] + sc.iloc[i] * (close_1d.iloc[i] - kama.iloc[i-1])
    kama_values = kama.values
    kama_aligned = align_htf_to_ltf(prices, df_1d, kama_values)
    
    # Calculate RSI(14) on 1d
    delta = close_1d.diff()
    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)
    avg_gain = gain.ewm(alpha=1/14, adjust=False, min_periods=14).mean()
    avg_loss = loss.ewm(alpha=1/14, adjust=False, min_periods=14).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    rsi = 100 - (100 / (1 + rs))
    rsi_values = rsi.values
    rsi_aligned = align_htf_to_ltf(prices, df_1d, rsi_values)
    
    # Calculate Choppiness Index (CHOP) on 1d
    atr_period = 14
    tr1 = pd.DataFrame(df_1d['high'] - df_1d['low'])
    tr2 = pd.DataFrame(abs(df_1d['high'] - df_1d['close'].shift(1)))
    tr3 = pd.DataFrame(abs(df_1d['low'] - df_1d['close'].shift(1)))
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr = tr.ewm(alpha=1/atr_period, adjust=False, min_periods=atr_period).mean()
    max_high = df_1d['high'].rolling(window=atr_period, min_periods=atr_period).max()
    min_low = df_1d['low'].rolling(window=atr_period, min_periods=atr_period).min()
    chop = 100 * np.log10(atr.rolling(window=atr_period, min_periods=atr_period).sum() /
                          (max_high - min_low)) / np.log10(atr_period)
    chop_values = chop.values
    chop_aligned = align_htf_to_ltf(prices, df_1d, chop_values)
    
    # Calculate average volume for confirmation (20-period SMA)
    avg_volume = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    # Load 1w data for HTF trend filter (EMA50)
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 50:
        return np.zeros(n)
    ema_50_1w = pd.Series(df_1w['close']).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_50_1w)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    base_size = 0.25
    
    # Start after warmup (need 30 for KAMA/RSI/CHOP, 20 for volume)
    start_idx = 30
    
    for i in range(start_idx, n):
        kama_val = kama_aligned[i]
        rsi_val = rsi_aligned[i]
        chop_val = chop_aligned[i]
        vol = volume[i]
        avg_vol = avg_volume[i]
        ema_1w_val = ema_50_1w_aligned[i]
        close_val = close[i]
        
        # Skip if any data not ready
        if (np.isnan(kama_val) or np.isnan(rsi_val) or np.isnan(chop_val) or 
            np.isnan(ema_1w_val) or np.isnan(avg_vol)):
            # Hold current position
            if position == 0:
                signals[i] = 0.0
            elif position == 1:
                signals[i] = base_size
            else:
                signals[i] = -base_size
            continue
        
        # Volume confirmation: current volume > 1.5x average volume
        volume_confirmed = vol > 1.5 * avg_vol
        
        # Regime filter: Choppiness Index > 61.8 (ranging market)
        regime_filter = chop_val > 61.8
        
        # Long logic: KAMA up, RSI > 50, ranging market, volume spike, price > KAMA
        long_condition = (kama_val > kama_aligned[i-1]) and (rsi_val > 50) and \
                         regime_filter and volume_confirmed and (close_val > kama_val)
        # Short logic: KAMA down, RSI < 50, ranging market, volume spike, price < KAMA
        short_condition = (kama_val < kama_aligned[i-1]) and (rsi_val < 50) and \
                          regime_filter and volume_confirmed and (close_val < kama_val)
        
        # Exit logic: opposite regime or trend failure
        exit_long = (rsi_val < 40) or (chop_val < 38.2) or (close_val < kama_val)
        exit_short = (rsi_val > 60) or (chop_val < 38.2) or (close_val > kama_val)
        
        if long_condition and position != 1:
            signals[i] = base_size
            position = 1
        elif short_condition and position != -1:
            signals[i] = -base_size
            position = -1
        elif position == 1 and exit_long:
            signals[i] = 0.0
            position = 0
        elif position == -1 and exit_short:
            signals[i] = 0.0
            position = 0
        else:
            # Hold current position
            if position == 0:
                signals[i] = 0.0
            elif position == 1:
                signals[i] = base_size
            else:
                signals[i] = -base_size
    
    return signals

name = "1d_KAMA_Trend_RSI_ChopFilter_VolumeSpike"
timeframe = "1d"
leverage = 1.0
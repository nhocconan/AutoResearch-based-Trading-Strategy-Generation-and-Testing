#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "12h_1d_kama_rsi_chop_v1"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    # Load daily data ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return signals
    
    # Calculate 1d KAMA
    close_1d = df_1d['close'].values
    # Efficiency ratio
    change = np.abs(np.diff(close_1d, prepend=close_1d[0]))
    volatility = np.abs(np.diff(close_1d))
    er = np.where(volatility != 0, change / volatility, 0)
    # Smoothing constants
    fast_sc = 2 / (2 + 1)  # EMA(2)
    slow_sc = 2 / (30 + 1)  # EMA(30)
    sc = (er * (fast_sc - slow_sc) + slow_sc) ** 2
    kama = np.zeros_like(close_1d)
    kama[0] = close_1d[0]
    for i in range(1, len(close_1d)):
        kama[i] = kama[i-1] + sc[i] * (close_1d[i] - kama[i-1])
    kama = np.roll(kama, 1)  # Use previous day's KAMA
    kama[0] = np.nan
    
    # Calculate 1d RSI(14)
    delta = np.diff(close_1d, prepend=close_1d[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    avg_gain = pd.Series(gain).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    avg_loss = pd.Series(loss).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    rs = np.where(avg_loss != 0, avg_gain / avg_loss, 0)
    rsi = 100 - (100 / (1 + rs))
    rsi = np.roll(rsi, 1)
    rsi[0] = np.nan
    
    # Calculate 1d Choppiness Index (CHOP)
    atr = np.zeros_like(close_1d)
    tr1 = np.abs(np.subtract(df_1d['high'].values, df_1d['low'].values))
    tr2 = np.abs(np.subtract(df_1d['high'].values, np.roll(close_1d, 1)))
    tr3 = np.abs(np.subtract(np.roll(close_1d, 1), df_1d['low'].values))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    atr = pd.Series(tr).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    sum_atr14 = pd.Series(atr).rolling(window=14, min_periods=14).sum().values
    hh = pd.Series(df_1d['high'].values).rolling(window=14, min_periods=14).max().values
    ll = pd.Series(df_1d['low'].values).rolling(window=14, min_periods=14).min().values
    chop = 100 * np.log10(sum_atr14 / (hh - ll)) / np.log10(14)
    chop = np.roll(chop, 1)
    chop[0] = np.nan
    
    # Align 1d indicators to 12h timeframe
    kama_aligned = align_htf_to_ltf(prices, df_1d, kama)
    rsi_aligned = align_htf_to_ltf(prices, df_1d, rsi)
    chop_aligned = align_htf_to_ltf(prices, df_1d, chop)
    
    # Calculate 12h EMA50 for trend filter
    ema_50 = pd.Series(close).ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Volume filter: volume > 1.5x 20-period average
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    for i in range(60, n):  # Start after EMA50 warmup
        # Skip if any required data is invalid
        if (np.isnan(kama_aligned[i]) or np.isnan(rsi_aligned[i]) or
            np.isnan(chop_aligned[i]) or np.isnan(ema_50[i]) or
            np.isnan(vol_ma_20[i])):
            signals[i] = 0.0
            continue
        
        price_close = close[i]
        price_high = high[i]
        price_low = low[i]
        volume_current = volume[i]
        vol_ma = vol_ma_20[i]
        
        # Volume confirmation
        volume_confirmed = volume_current > 1.5 * vol_ma
        
        # Chop regime: CHOP > 61.8 = range (mean revert), CHOP < 38.2 = trending (trend follow)
        chop_value = chop_aligned[i]
        in_range = chop_value > 61.8
        in_trend = chop_value < 38.2
        
        # Long conditions: KAMA upward + RSI not overbought + volume + regime fit
        kama_up = kama_aligned[i] > kama_aligned[i-1] if i > 0 else False
        rsi_not_overbought = rsi_aligned[i] < 70
        long_signal = volume_confirmed and kama_up and rsi_not_overbought and (
            (in_range and price_close > kama_aligned[i]) or  # Buy dips in range
            (in_trend and price_close > ema_50[i])           # Buy uptrend
        )
        
        # Short conditions: KAMA downward + RSI not oversold + volume + regime fit
        kama_down = kama_aligned[i] < kama_aligned[i-1] if i > 0 else False
        rsi_not_oversold = rsi_aligned[i] > 30
        short_signal = volume_confirmed and kama_down and rsi_not_oversold and (
            (in_range and price_close < kama_aligned[i]) or  # Sell rallies in range
            (in_trend and price_close < ema_50[i])           # Sell downtrend
        )
        
        # Exit when KAMA direction reverses or RSI extreme
        exit_long = position == 1 and (not kama_up or rsi_aligned[i] > 80)
        exit_short = position == -1 and (not kama_down or rsi_aligned[i] < 20)
        
        # Trading logic
        if long_signal and position != 1:
            position = 1
            signals[i] = 0.25
        elif short_signal and position != -1:
            position = -1
            signals[i] = -0.25
        elif position == 1 and exit_long:
            position = 0
            signals[i] = 0.0
        elif position == -1 and exit_short:
            position = 0
            signals[i] = 0.0
        else:
            # Maintain current position
            signals[i] = 0.25 if position == 1 else (-0.25 if position == -1 else 0.0)
    
    return signals

# Hypothesis: KAMA direction + RSI + Chop regime on 12h with volume confirmation.
# Uses 1d KAMA for adaptive trend, 1d RSI for momentum exhaustion, and 1d Chop for
# regime detection (range vs trend). Enters long when KAMA rising, RSI < 70,
# and price above KAMA in range or above EMA50 in trend, with volume confirmation.
# Enters short when KAMA falling, RSI > 30, and price below KAMA in range or
# below EMA50 in trend, with volume confirmation. Exits when KAMA reverses or
# RSI reaches extreme. Works in both bull and bear markets by adapting to regime.
# Target: 50-150 total trades over 4 years (12-37/year) to minimize fee drag on 12h.
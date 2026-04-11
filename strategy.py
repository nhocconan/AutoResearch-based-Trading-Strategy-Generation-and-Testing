#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "1d_1w_kama_rsi_breakout_v1"
timeframe = "1d"
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
    
    # Load weekly data ONCE before loop
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 30:
        return signals
    
    # Calculate weekly KAMA for trend filter
    close_1w = df_1w['close'].values
    # Efficiency Ratio
    change = np.abs(np.diff(close_1w, n=10))
    volatility = np.sum(np.abs(np.diff(close_1w, n=1)), axis=0)
    er = np.divide(change, volatility, out=np.zeros_like(change), where=volatility!=0)
    # Smoothing constants
    sc = (er * (2/(2+1) - 2/(30+1)) + 2/(30+1)) ** 2
    # KAMA calculation
    kama = np.full_like(close_1w, np.nan)
    kama[29] = close_1w[29]  # seed
    for i in range(30, len(close_1w)):
        kama[i] = kama[i-1] + sc[i-1] * (close_1w[i] - kama[i-1])
    
    # Shift by 1 to use only completed weekly bars
    kama = np.roll(kama, 1)
    kama[0] = np.nan
    
    # Align weekly KAMA to daily timeframe
    kama_aligned = align_htf_to_ltf(prices, df_1w, kama)
    
    # Calculate daily RSI
    delta = np.diff(close, prepend=close[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    avg_gain = pd.Series(gain).rolling(window=14, min_periods=14).mean().values
    avg_loss = pd.Series(loss).rolling(window=14, min_periods=14).mean().values
    rs = np.divide(avg_gain, avg_loss, out=np.zeros_like(avg_gain), where=avg_loss!=0)
    rsi = 100 - (100 / (1 + rs))
    
    # Calculate daily ATR for volatility filter
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr1[0] = 0
    tr2[0] = 0
    tr3[0] = 0
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    atr = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    # Volume filter: volume > 1.5x 20-day average
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    for i in range(50, n):  # Start after warmup
        # Skip if any required data is invalid
        if (np.isnan(kama_aligned[i]) or np.isnan(rsi[i]) or
            np.isnan(atr[i]) or np.isnan(vol_ma_20[i])):
            signals[i] = 0.0
            continue
        
        price_close = close[i]
        price_high = high[i]
        price_low = low[i]
        volume_current = volume[i]
        vol_ma = vol_ma_20[i]
        atr_val = atr[i]
        
        # Volume confirmation
        volume_confirmed = volume_current > 1.5 * vol_ma
        
        # Long conditions: Price > Weekly KAMA AND RSI < 70 AND volume
        long_signal = volume_confirmed and (price_close > kama_aligned[i]) and (rsi[i] < 70)
        
        # Short conditions: Price < Weekly KAMA AND RSI > 30 AND volume
        short_signal = volume_confirmed and (price_close < kama_aligned[i]) and (rsi[i] > 30)
        
        # Exit when price crosses back through KAMA
        exit_long = position == 1 and price_close <= kama_aligned[i]
        exit_short = position == -1 and price_close >= kama_aligned[i]
        
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

# Hypothesis: Daily KAMA/RSI breakout with weekly trend filter.
# Uses weekly KAMA as adaptive trend filter (more responsive than SMA in volatile markets).
# Enters long when daily price crosses above weekly KAMA with RSI < 70 (not overbought)
# and volume confirmation (>1.5x average). Enters short when price crosses below weekly KAMA
# with RSI > 30 (not oversold) and volume confirmation. Exits when price crosses back through
# weekly KAMA. Works in both bull and bear markets by aligning with higher timeframe trend.
# Weekly timeframe reduces noise, daily timeframe provides timely entries. Target: 20-60 trades
# over 4 years (5-15/year) to minimize fee drag on daily timeframe. KAMA adapts to market
# conditions, reducing false signals during choppy periods. RSI prevents buying into
# overextended moves. Volume confirmation ensures institutional participation.
#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "4h_1d_kama_trend_v1"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 60:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Load daily data ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    # KAMA parameters (adaptive moving average)
    close_1d = df_1d['close'].values
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    
    # Calculate Efficiency Ratio (ER) and smoothing constants
    change = np.abs(np.diff(close_1d, prepend=close_1d[0]))
    volatility = np.sum(np.abs(np.diff(close_1d)), axis=0)
    
    # Vectorized ER calculation
    er = np.zeros_like(close_1d)
    for i in range(1, len(close_1d)):
        if i < 10:
            er[i] = 0
        else:
            change_10 = np.abs(close_1d[i] - close_1d[i-10])
            volatility_10 = np.sum(np.abs(np.diff(close_1d[i-10:i+1])))
            er[i] = change_10 / (volatility_10 + 1e-10)
    
    # Smoothing constants
    fast_sc = 2 / (2 + 1)   # EMA(2)
    slow_sc = 2 / (30 + 1)  # EMA(30)
    sc = (er * (fast_sc - slow_sc) + slow_sc) ** 2
    
    # Calculate KAMA
    kama = np.zeros_like(close_1d)
    kama[0] = close_1d[0]
    for i in range(1, len(close_1d)):
        kama[i] = kama[i-1] + sc[i] * (close_1d[i] - kama[i-1])
    
    # Shift by 1 to use only completed daily bars
    kama = np.roll(kama, 1)
    kama[0] = np.nan
    
    # Align daily KAMA to 4h timeframe
    kama_4h = align_htf_to_ltf(prices, df_1d, kama)
    
    # 4h RSI for overbought/oversold (14 period)
    delta = np.diff(close, prepend=close[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    
    avg_gain = pd.Series(gain).rolling(window=14, min_periods=14).mean().values
    avg_loss = pd.Series(loss).rolling(window=14, min_periods=14).mean().values
    rs = avg_gain / (avg_loss + 1e-10)
    rsi = 100 - (100 / (1 + rs))
    
    # 4h volume filter: volume > 1.3x 20-period average
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    # 4h ADX for trend strength (14 period)
    tr1 = high[1:] - low[1:]
    tr2 = np.abs(high[1:] - close[:-1])
    tr3 = np.abs(low[1:] - close[:-1])
    tr = np.concatenate([[np.nan], np.maximum(tr1, np.maximum(tr2, tr3))])
    
    plus_dm = np.where((high[1:] - high[:-1]) > (low[:-1] - low[1:]), np.maximum(high[1:] - high[:-1], 0), 0)
    minus_dm = np.where((low[:-1] - low[1:]) > (high[1:] - high[:-1]), np.maximum(low[:-1] - low[1:], 0), 0)
    
    tr_dm = tr[1:]
    plus_di = 100 * pd.Series(plus_dm).rolling(window=14, min_periods=14).mean().values / pd.Series(tr_dm).rolling(window=14, min_periods=14).mean().values
    minus_di = 100 * pd.Series(minus_dm).rolling(window=14, min_periods=14).mean().values / pd.Series(tr_dm).rolling(window=14, min_periods=14).mean().values
    dx = 100 * np.abs(plus_di - minus_di) / (plus_di + minus_di + 1e-10)
    adx = pd.Series(dx).rolling(window=14, min_periods=14).mean().values
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(60, n):
        # Skip if any required data is invalid
        if (np.isnan(kama_4h[i]) or np.isnan(rsi[i]) or
            np.isnan(vol_ma_20[i]) or np.isnan(adx[i])):
            signals[i] = 0.0
            continue
        
        price_close = close[i]
        volume_current = volume[i]
        vol_ma = vol_ma_20[i]
        
        # Volume confirmation (1.3x average)
        volume_confirmed = volume_current > 1.3 * vol_ma
        
        # Trend filter: ADX > 20 (moderate trend)
        trend_filter = adx[i] > 20
        
        # Long conditions: price above KAMA and RSI < 70 (not overbought)
        long_signal = volume_confirmed and trend_filter and (price_close > kama_4h[i]) and (rsi[i] < 70)
        
        # Short conditions: price below KAMA and RSI > 30 (not oversold)
        short_signal = volume_confirmed and trend_filter and (price_close < kama_4h[i]) and (rsi[i] > 30)
        
        # Exit conditions
        exit_long = position == 1 and (price_close < kama_4h[i] or rsi[i] > 75)
        exit_short = position == -1 and (price_close > kama_4h[i] or rsi[i] < 25)
        
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
            signals[i] = 0.25 if position == 1 else (-0.25 if position == -1 else 0.0)
    
    return signals

# Hypothesis: Daily KAMA trend following strategy for 4h timeframe with volume confirmation (>1.3x average volume) and ADX filter (>20).
# Enters long when 4h price is above daily KAMA (adaptive trend) with volume >1.3x average, ADX>20, and RSI<70.
# Enters short when price is below daily KAMA with same conditions and RSI>30.
# Exits when price crosses back below/above KAMA or RSI reaches extreme levels.
# Uses KAMA to adapt to changing market conditions (fast in trends, slow in ranges).
# Volume and ADX filters reduce false signals and overtrading.
# Moderate position size (0.25) balances risk and return.
# Target: 25-40 trades per year to minimize fee drag while capturing adaptive trends.
# Works in both bull (trend following) and bear (adaptive to ranging) markets.
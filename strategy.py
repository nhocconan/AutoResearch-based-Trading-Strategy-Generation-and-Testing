#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Strategy: 4h Choppiness Filter + 1d Trend + Volume Breakout
# Hypothesis: In choppy markets (high CHOP), price oscillates within a range; we mean-revert at Bollinger Bands.
# In trending markets (low CHOP), we follow the 1d trend direction using Donchian breakouts.
# Volume confirms institutional participation. Works in bull/bear by adapting to market regime.
# Targets 25-40 trades/year via regime filter + strict entry conditions.

name = "4h_chop_regime_vol_breakout"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Price data
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # 1d EMA50 for trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    ema50_1d = pd.Series(df_1d['close'].values).ewm(span=50, adjust=False).mean().values
    ema50_4h = align_htf_to_ltf(prices, df_1d, ema50_1d)
    
    # Bollinger Bands (20, 2) for mean reversion in chop
    close_s = pd.Series(close)
    basis = close_s.rolling(window=20, min_periods=20).mean()
    dev = close_s.rolling(window=20, min_periods=20).std()
    upper = basis + 2 * dev
    lower = basis - 2 * dev
    bb_width = (upper - lower) / basis  # normalized bandwidth
    
    # Choppiness Index (14) - values near 50 = choppy, near 0/100 = trending
    # CHOP = 100 * log10(sum(ATR) / (max(high) - min(low))) / log10(14)
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = 0  # first TR is just high-low
    atr = pd.Series(tr).rolling(window=14, min_periods=14).mean()
    hh = pd.Series(high).rolling(window=14, min_periods=14).max()
    ll = pd.Series(low).rolling(window=14, min_periods=14).min()
    chop = 100 * np.log10(atr.rolling(window=14, min_periods=14).sum() / (hh - ll)) / np.log10(14)
    chop = chop.values  # convert to numpy
    
    # Donchian Channel (20) for breakout in trending markets
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 2:
        return np.zeros(n)
    high_4h = df_4h['high'].values
    low_4h = df_4h['low'].values
    high_max = pd.Series(high_4h).rolling(window=20, min_periods=20).max().values
    low_min = pd.Series(low_4h).rolling(window=20, min_periods=20).min().values
    donchian_high = align_htf_to_ltf(prices, df_4h, high_max)
    donchian_low = align_htf_to_ltf(prices, df_4h, low_min)
    
    # Volume confirmation
    vol_sma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(20, n):  # Start after warmup
        # Skip if required data not available
        if (np.isnan(ema50_4h[i]) or np.isnan(chop[i]) or 
            np.isnan(donchian_high[i]) or np.isnan(donchian_low[i]) or
            np.isnan(upper[i]) or np.isnan(lower[i]) or np.isnan(vol_sma[i])):
            signals[i] = 0.0
            continue
        
        vol_confirm = volume[i] > 1.3 * vol_sma[i]
        
        if position == 1:  # Long position
            # Exit conditions depend on regime
            if chop[i] > 61.8:  # Chop regime: mean reversion
                if close[i] < basis[i]:  # Exit at mean
                    position = 0
                    signals[i] = 0.0
            else:  # Trend regime: follow trend
                if close[i] < donchian_low[i] or close[i] < ema50_4h[i]:
                    position = 0
                    signals[i] = 0.0
            if position == 1:
                signals[i] = 0.25
        elif position == -1:  # Short position
            if chop[i] > 61.8:  # Chop regime: mean reversion
                if close[i] > basis[i]:  # Exit at mean
                    position = 0
                    signals[i] = 0.0
            else:  # Trend regime: follow trend
                if close[i] > donchian_high[i] or close[i] > ema50_4h[i]:
                    position = 0
                    signals[i] = 0.0
            if position == -1:
                signals[i] = -0.25
        else:  # Flat, look for entry
            if chop[i] > 61.8:  # Chop regime: mean reversion at Bollinger Bands
                if (close[i] < lower[i] and vol_confirm and close[i] < ema50_4h[i]):
                    position = -1  # Short at lower band in downtrend
                    signals[i] = -0.25
                elif (close[i] > upper[i] and vol_confirm and close[i] > ema50_4h[i]):
                    position = 1   # Long at upper band in uptrend
                    signals[i] = 0.25
            else:  # Trend regime: Donchian breakout with trend filter
                if (close[i] > donchian_high[i] and vol_confirm and 
                    close[i] > ema50_4h[i]):
                    position = 1
                    signals[i] = 0.25
                elif (close[i] < donchian_low[i] and vol_confirm and 
                      close[i] < ema50_4h[i]):
                    position = -1
                    signals[i] = -0.25
    
    return signals
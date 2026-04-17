# 2025-07-07: 4h Donchian Breakout with Volume Confirmation and ADX Trend Filter
# Hypothesis: Donchian(20) breakouts capture breakout momentum in both bull and bear markets.
# Volume > 1.5x 20-bar average confirms institutional participation.
# ADX > 25 ensures we only trade in trending regimes, reducing false breakouts in ranging markets.
# Fixed position size 0.25 to balance risk and reward, targeting 20-50 trades/year.
# Uses 1d EMA50 as additional trend filter to avoid counter-trend trades.

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get daily data for EMA50 trend filter
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate 1d EMA50 for trend filter
    ema_50 = pd.Series(df_1d['close']).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_4h = align_htf_to_ltf(prices, df_1d, ema_50)
    
    # Calculate ADX(14) on 4h for trend strength
    plus_dm = np.where((high[1:] - high[:-1]) > (low[:-1] - low[1:]), np.maximum(high[1:] - high[:-1], 0), 0)
    minus_dm = np.where((low[:-1] - low[1:]) > (high[1:] - high[:-1]), np.maximum(low[:-1] - low[1:], 0), 0)
    tr = np.maximum(high[1:] - low[1:], np.absolute(high[1:] - close[:-1]), np.absolute(low[1:] - close[:-1]))
    plus_di = 100 * (pd.Series(plus_dm).ewm(alpha=1/14, adjust=False).mean() / pd.Series(tr).ewm(alpha=1/14, adjust=False).mean())
    minus_di = 100 * (pd.Series(minus_dm).ewm(alpha=1/14, adjust=False).mean() / pd.Series(tr).ewm(alpha=1/14, adjust=False).mean())
    dx = (np.absolute(plus_di - minus_di) / (plus_di + minus_di)) * 100
    adx = pd.Series(dx).ewm(alpha=1/14, adjust=False).mean().values
    # Prepend NaN for alignment
    adx = np.concatenate([np.full(1, np.nan), adx.values])
    
    # Donchian channels (20-period)
    donchian_high = pd.Series(high).rolling(window=20, min_periods=20).max().values
    donchian_low = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # Volume confirmation: 20-period volume MA
    volume_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # -1: short, 0: flat, 1: long
    
    start_idx = 50  # warmup for all indicators
    
    for i in range(start_idx, n):
        if (np.isnan(donchian_high[i]) or np.isnan(donchian_low[i]) or
            np.isnan(ema_50_4h[i]) or np.isnan(adx[i]) or np.isnan(volume_ma_20[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        vol = volume[i]
        vol_ma = volume_ma_20[i]
        
        if position == 0:
            # Long: break above Donchian high with volume confirmation and ADX > 25
            if price > donchian_high[i] and vol > 1.5 * vol_ma and adx[i] > 25 and price > ema_50_4h[i]:
                signals[i] = 0.25
                position = 1
            # Short: break below Donchian low with volume confirmation and ADX > 25
            elif price < donchian_low[i] and vol > 1.5 * vol_ma and adx[i] > 25 and price < ema_50_4h[i]:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Long exit: price returns to Donchian low or ADX drops below 20
            if price < donchian_low[i] or adx[i] < 20:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: price returns to Donchian high or ADX drops below 20
            if price > donchian_high[i] or adx[i] < 20:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "4h_DonchianBreakout_Volume_ADX"
timeframe = "4h"
leverage = 1.0
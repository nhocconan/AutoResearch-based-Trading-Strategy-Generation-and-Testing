#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1h strategy using 4h Donchian channel breakout with 1d EMA200 trend filter and volume confirmation
# Long when price breaks above 4h Donchian upper (20) + 1d EMA200 uptrend + volume > 1.5x 20-period avg
# Short when price breaks below 4h Donchian lower (20) + 1d EMA200 downtrend + volume > 1.5x 20-period avg
# Uses discrete position sizing (0.20) to minimize fee drag and control drawdown.
# 1d EMA200 provides strong trend filter reducing whipsaws in both bull and bear markets.
# Volume threshold (1.5x) targets ~15-35 trades/year to minimize fee drag on 1h timeframe.
# 4h Donchian channels provide structure while 1h timeframe allows precise entry timing.

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Precompute session hours (08-20 UTC) for filter
    hours = pd.DatetimeIndex(prices['open_time']).hour
    in_session = (hours >= 8) & (hours <= 20)
    
    # Get 4h HTF data once before loop
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 50:
        return np.zeros(n)
    
    # === 4h Indicator: Donchian Channel (20) ===
    high_4h = df_4h['high'].values
    low_4h = df_4h['low'].values
    donchian_high_20 = pd.Series(high_4h).rolling(window=20, min_periods=20).max().values
    donchian_low_20 = pd.Series(low_4h).rolling(window=20, min_periods=20).min().values
    donchian_high_20_aligned = align_htf_to_ltf(prices, df_4h, donchian_high_20)
    donchian_low_20_aligned = align_htf_to_ltf(prices, df_4h, donchian_low_20)
    
    # Get 1d HTF data once before loop
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # === 1d Indicator: EMA200 ===
    close_1d = df_1d['close'].values
    ema_200_1d = pd.Series(close_1d).ewm(span=200, adjust=False, min_periods=200).mean().values
    ema_200_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_200_1d)
    
    # Volume SMA for confirmation (using 20-period)
    vol_sma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    
    # Warmup: ensure all indicators are valid
    warmup = max(200, 20) + 5  # EMA200 + donchian(20) + volume(20) + buffer
    
    for i in range(warmup, n):
        # Skip if outside trading session (08-20 UTC)
        if not in_session[i]:
            signals[i] = 0.0
            continue
        
        # Skip if any required data is NaN
        if (np.isnan(donchian_high_20_aligned[i]) or np.isnan(donchian_low_20_aligned[i]) or
            np.isnan(ema_200_1d_aligned[i]) or np.isnan(vol_sma_20[i])):
            signals[i] = 0.0
            continue
        
        # Volume filter: current volume > 1.5x 20-period volume SMA
        vol_confirm = volume[i] > (vol_sma_20[i] * 1.5)
        
        # === LONG CONDITIONS ===
        # 1. Price breaks above 4h Donchian upper (20)
        # 2. 1d EMA200 uptrend (close > EMA200)
        # 3. Volume confirmation
        if (close[i] > donchian_high_20_aligned[i]) and \
           (close[i] > ema_200_1d_aligned[i]) and vol_confirm:
            signals[i] = 0.20
        
        # === SHORT CONDITIONS ===
        # 1. Price breaks below 4h Donchian lower (20)
        # 2. 1d EMA200 downtrend (close < EMA200)
        # 3. Volume confirmation
        elif (close[i] < donchian_low_20_aligned[i]) and \
             (close[i] < ema_200_1d_aligned[i]) and vol_confirm:
            signals[i] = -0.20
        
        else:
            signals[i] = 0.0  # flat
    
    return signals

name = "1h_4hDonchian20_1dEMA200_Volume_Filter_v1"
timeframe = "1h"
leverage = 1.0
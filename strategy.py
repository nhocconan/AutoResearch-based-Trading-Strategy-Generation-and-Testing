#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1d Donchian(20) breakout with 1w EMA34 trend filter and volume confirmation
# Long when price breaks above 20-period Donchian high + 1w EMA34 uptrend + volume > 2.0x 20-period avg
# Short when price breaks below 20-period Donchian low + 1w EMA34 downtrend + volume > 2.0x 20-period avg
# Uses discrete position sizing (0.25) to minimize fee drag and control drawdown.
# 1w EMA34 provides strong trend filter reducing whipsaws in both bull and bear markets.
# Volume threshold (2.0x) targets ~15-25 trades/year to minimize fee drag on 1d timeframe.
# Donchian channels calculated from HTF 1d data.

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1d HTF data once before loop (primary timeframe is 1d, so this is same as prices but we use it for HTF indicators)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # === 1d Indicator: Donchian Channel (20) ===
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    
    # Calculate Donchian high/low (20-period)
    donchian_high = pd.Series(high_1d).rolling(window=20, min_periods=20).max().values
    donchian_low = pd.Series(low_1d).rolling(window=20, min_periods=20).min().values
    
    # Align Donchian levels to 1d bars (no shift needed as we're using completed bars)
    donchian_high_aligned = align_htf_to_ltf(prices, df_1d, donchian_high)
    donchian_low_aligned = align_htf_to_ltf(prices, df_1d, donchian_low)
    
    # === 1w HTF Indicator: EMA34 ===
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 34:
        return np.zeros(n)
    
    close_1w = df_1w['close'].values
    ema_34_1w = pd.Series(close_1w).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_34_1w)
    
    # Volume SMA for confirmation (using 20-period)
    vol_sma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    
    # Warmup: ensure all indicators are valid
    warmup = max(34, 20) + 5  # EMA34 + Donchian(20) + volume(20) + buffer
    
    for i in range(warmup, n):
        # Skip if any required data is NaN
        if (np.isnan(donchian_high_aligned[i]) or np.isnan(donchian_low_aligned[i]) or
            np.isnan(ema_34_1w_aligned[i]) or np.isnan(vol_sma_20[i])):
            signals[i] = 0.0
            continue
        
        # Volume filter: current volume > 2.0x 20-period volume SMA
        vol_confirm = volume[i] > (vol_sma_20[i] * 2.0)
        
        # === LONG CONDITIONS ===
        # 1. Price breaks above Donchian high (close > Donchian high)
        # 2. 1w EMA34 uptrend (close > EMA34)
        # 3. Volume confirmation
        if (close[i] > donchian_high_aligned[i]) and \
           (close[i] > ema_34_1w_aligned[i]) and vol_confirm:
            signals[i] = 0.25
        
        # === SHORT CONDITIONS ===
        # 1. Price breaks below Donchian low (close < Donchian low)
        # 2. 1w EMA34 downtrend (close < EMA34)
        # 3. Volume confirmation
        elif (close[i] < donchian_low_aligned[i]) and \
             (close[i] < ema_34_1w_aligned[i]) and vol_confirm:
            signals[i] = -0.25
        
        else:
            signals[i] = 0.0  # flat
    
    return signals

name = "1d_Donchian20_1wEMA34_Volume_Filter_v1"
timeframe = "1d"
leverage = 1.0
#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Elder Ray Bull/Bear Power with 1d EMA34 trend filter and volume confirmation
# Long when Bull Power > 0 (close > EMA13) + Bear Power < 0 (high < EMA13) + 1d EMA34 uptrend + volume > 1.5x 20-period avg
# Short when Bear Power < 0 (high < EMA13) + Bull Power < 0 (low > EMA13) + 1d EMA34 downtrend + volume > 1.5x 20-period avg
# Uses discrete position sizing (0.25) to minimize fee drag and control drawdown.
# 1d EMA34 provides strong trend filter reducing whipsaws in both bull and bear markets.
# Volume threshold (1.5x) targets ~50-100 trades/year to minimize fee drag on 6h timeframe.
# Elder Ray measures bull/bear power relative to EMA13, providing institutional-grade trend strength assessment.

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
    
    # Get 1d HTF data once before loop
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 34:
        return np.zeros(n)
    
    # === 1d Indicator: EMA34 ===
    close_1d = df_1d['close'].values
    ema_34_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # === 6h Indicators: EMA13 for Elder Ray ===
    ema_13 = pd.Series(close).ewm(span=13, adjust=False, min_periods=13).mean().values
    
    # Calculate Elder Ray components
    bull_power = close - ema_13  # Bull Power = Close - EMA13
    bear_power = high - ema_13   # Bear Power = High - EMA13 (note: some definitions use Low, but High works better for shorts)
    
    # Volume SMA for confirmation (using 20-period)
    vol_sma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    
    # Warmup: ensure all indicators are valid
    warmup = max(34, 20) + 5  # EMA34 + volume(20) + buffer
    
    for i in range(warmup, n):
        # Skip if outside trading session (08-20 UTC)
        if not in_session[i]:
            signals[i] = 0.0
            continue
        
        # Skip if any required data is NaN
        if (np.isnan(ema_34_1d_aligned[i]) or np.isnan(bull_power[i]) or 
            np.isnan(bear_power[i]) or np.isnan(vol_sma_20[i])):
            signals[i] = 0.0
            continue
        
        # Volume filter: current volume > 1.5x 20-period volume SMA
        vol_confirm = volume[i] > (vol_sma_20[i] * 1.5)
        
        # === LONG CONDITIONS ===
        # 1. Bull Power > 0 (close > EMA13) - bulls in control
        # 2. Bear Power < 0 (high < EMA13) - bears weak (no new highs above EMA13)
        # 3. 1d EMA34 uptrend (close > EMA34)
        # 4. Volume confirmation
        if (bull_power[i] > 0) and \
           (bear_power[i] < 0) and \
           (close[i] > ema_34_1d_aligned[i]) and vol_confirm:
            signals[i] = 0.25
        
        # === SHORT CONDITIONS ===
        # 1. Bear Power < 0 (high < EMA13) - bears in control (no new highs above EMA13)
        # 2. Bull Power < 0 (low > EMA13) - bulls weak (no new lows below EMA13)
        # 3. 1d EMA34 downtrend (close < EMA34)
        # 4. Volume confirmation
        elif (bear_power[i] < 0) and \
             (bull_power[i] < 0) and \
             (close[i] < ema_34_1d_aligned[i]) and vol_confirm:
            signals[i] = -0.25
        
        else:
            signals[i] = 0.0  # flat
    
    return signals

name = "6h_ElderRay_BullBearPower_1dEMA34_Volume_Filter_v1"
timeframe = "6h"
leverage = 1.0
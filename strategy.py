#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1d Donchian(20) breakout with 1w EMA(34) trend filter and volume confirmation
# Long when price breaks above 1d Donchian upper band + 1w EMA34 uptrend + volume > 1.5x 20-period avg
# Short when price breaks below 1d Donchian lower band + 1w EMA34 downtrend + volume > 1.5x 20-period avg
# Uses discrete position sizing (0.25) to balance return and drawdown control.
# 1w EMA34 provides strong trend filter reducing whipsaws in both bull and bear markets.
# Volume threshold (1.5x) targets ~15-25 trades/year to minimize fee drag on 1d timeframe.

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
    
    # Get 1w HTF data once before loop
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 40:
        return np.zeros(n)
    
    # === 1w Indicator: EMA(34) ===
    close_1w = df_1w['close'].values
    ema_34_1w = pd.Series(close_1w).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_34_1w)
    
    # === 1d Donchian Channel (20-period) ===
    period = 20
    highest_high = pd.Series(high).rolling(window=period, min_periods=period).max().values
    lowest_low = pd.Series(low).rolling(window=period, min_periods=period).min().values
    upper_band = highest_high
    lower_band = lowest_low
    
    # Volume SMA for confirmation (using 20-period)
    vol_sma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    
    # Warmup: ensure all indicators are valid
    warmup = max(period, 20, 34) + 5  # Donchian(20) + volume(20) + EMA(34) buffer
    
    for i in range(warmup, n):
        # Skip if outside trading session (08-20 UTC)
        if not in_session[i]:
            signals[i] = 0.0
            continue
        
        # Volume filter: current volume > 1.5x 20-period volume SMA
        vol_confirm = volume[i] > (vol_sma_20[i] * 1.5)
        
        # Skip if any required data is NaN
        if (np.isnan(upper_band[i]) or np.isnan(lower_band[i]) or
            np.isnan(ema_34_1w_aligned[i]) or np.isnan(vol_sma_20[i])):
            signals[i] = 0.0
            continue
        
        # === LONG CONDITIONS ===
        # 1. Price breaks above upper band (close > upper_band)
        # 2. 1w EMA34 uptrend (price > EMA)
        # 3. Volume confirmation
        if (close[i] > upper_band[i]) and \
           (close[i] > ema_34_1w_aligned[i]) and vol_confirm:
            signals[i] = 0.25
        
        # === SHORT CONDITIONS ===
        # 1. Price breaks below lower band (close < lower_band)
        # 2. 1w EMA34 downtrend (price < EMA)
        # 3. Volume confirmation
        elif (close[i] < lower_band[i]) and \
             (close[i] < ema_34_1w_aligned[i]) and vol_confirm:
            signals[i] = -0.25
        
        else:
            signals[i] = 0.0  # flat
    
    return signals

name = "1d_Donchian20_1wEMA34_Volume_Filter_v1"
timeframe = "1d"
leverage = 1.0
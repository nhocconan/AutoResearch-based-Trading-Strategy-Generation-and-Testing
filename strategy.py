#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h Williams Alligator + Elder Ray combo with 1d EMA50 trend filter and volume confirmation
# Long when: Alligator bullish (jaw < teeth < lips) + Elder Ray bullish (bull power > 0) + price > 1d EMA50 + volume > 1.5x 20-period avg
# Short when: Alligator bearish (jaw > teeth > lips) + Elder Ray bearish (bear power < 0) + price < 1d EMA50 + volume > 1.5x 20-period avg
# Uses discrete position sizing (0.25) to minimize fee drag and control drawdown.
# 1d EMA50 provides strong trend filter reducing whipsaws in both bull and bear markets.
# Volume threshold (1.5x) targets ~20-40 trades/year to minimize fee drag on 12h timeframe.
# Williams Alligator: jaw=SMA(13,8), teeth=SMA(8,5), lips=SMA(5,3)
# Elder Ray: bull power = high - EMA(13), bear power = low - EMA(13)

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
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # === 1d Indicator: EMA50 ===
    close_1d = df_1d['close'].values
    ema_50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # === Williams Alligator (12h timeframe) ===
    # Jaw: SMA(13, 8 periods offset)
    jaw = pd.Series(close).rolling(window=13, min_periods=13).mean().shift(8).values
    # Teeth: SMA(8, 5 periods offset)
    teeth = pd.Series(close).rolling(window=8, min_periods=8).mean().shift(5).values
    # Lips: SMA(5, 3 periods offset)
    lips = pd.Series(close).rolling(window=5, min_periods=5).mean().shift(3).values
    
    # === Elder Ray (1d timeframe) ===
    ema_13_1d = pd.Series(close_1d).ewm(span=13, adjust=False, min_periods=13).mean().values
    ema_13_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_13_1d)
    bull_power = high - ema_13_1d_aligned
    bear_power = low - ema_13_1d_aligned
    
    # Volume SMA for confirmation (using 20-period)
    vol_sma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    
    # Warmup: ensure all indicators are valid
    warmup = max(50, 20, 13) + 5  # EMA50 + volume(20) + Alligator/Elder Ray + buffer
    
    for i in range(warmup, n):
        # Skip if outside trading session (08-20 UTC)
        if not in_session[i]:
            signals[i] = 0.0
            continue
        
        # Skip if any required data is NaN
        if (np.isnan(jaw[i]) or np.isnan(teeth[i]) or np.isnan(lips[i]) or
            np.isnan(ema_50_1d_aligned[i]) or np.isnan(bull_power[i]) or np.isnan(bear_power[i]) or
            np.isnan(vol_sma_20[i])):
            signals[i] = 0.0
            continue
        
        # Volume filter: current volume > 1.5x 20-period volume SMA
        vol_confirm = volume[i] > (vol_sma_20[i] * 1.5)
        
        # === LONG CONDITIONS ===
        # 1. Alligator bullish: jaw < teeth < lips
        # 2. Elder Ray bullish: bull power > 0
        # 3. Price above 1d EMA50
        # 4. Volume confirmation
        if (jaw[i] < teeth[i]) and (teeth[i] < lips[i]) and \
           (bull_power[i] > 0) and \
           (close[i] > ema_50_1d_aligned[i]) and vol_confirm:
            signals[i] = 0.25
        
        # === SHORT CONDITIONS ===
        # 1. Alligator bearish: jaw > teeth > lips
        # 2. Elder Ray bearish: bear power < 0
        # 3. Price below 1d EMA50
        # 4. Volume confirmation
        elif (jaw[i] > teeth[i]) and (teeth[i] > lips[i]) and \
             (bear_power[i] < 0) and \
             (close[i] < ema_50_1d_aligned[i]) and vol_confirm:
            signals[i] = -0.25
        
        else:
            signals[i] = 0.0  # flat
    
    return signals

name = "12h_Alligator_ElderRay_1dEMA50_Volume_Filter_v1"
timeframe = "12h"
leverage = 1.0
#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h Williams Alligator + 1d EMA50 trend filter + volume spike
# Long when: Alligator jaws < teeth < lips (bullish alignment) + price > lips + 1d EMA50 uptrend + volume > 2.0x 20-period avg
# Short when: Alligator jaws > teeth > lips (bearish alignment) + price < lips + 1d EMA50 downtrend + volume > 2.0x 20-period avg
# Williams Alligator uses SMAs of median price with specific periods (13,8,5) and shifts (8,5,3)
# This strategy targets 12-30 trades/year on 12h timeframe to avoid fee drag while capturing strong trends.
# The 1d EMA50 filter reduces whipsaws in both bull and bear markets by ensuring alignment with higher timeframe trend.
# Volume threshold (2.0x) ensures entries occur only during significant participation, reducing false breakouts.

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
    
    # === Williams Alligator (using median price) ===
    # Median price = (high + low) / 2
    median_price = (high + low) / 2.0
    
    # Alligator lines: SMAs of median price with specific periods and shifts
    # Jaw: SMA(13) shifted 8 bars
    # Teeth: SMA(8) shifted 5 bars  
    # Lips: SMA(5) shifted 3 bars
    jaw = pd.Series(median_price).rolling(window=13, min_periods=13).mean().shift(8).values
    teeth = pd.Series(median_price).rolling(window=8, min_periods=8).mean().shift(5).values
    lips = pd.Series(median_price).rolling(window=5, min_periods=5).mean().shift(3).values
    
    # Volume SMA for confirmation (using 20-period)
    vol_sma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    
    # Warmup: ensure all indicators are valid
    warmup = max(50, 13+8, 20) + 5  # EMA50 + Alligator jaw shift + volume + buffer
    
    for i in range(warmup, n):
        # Skip if outside trading session (08-20 UTC)
        if not in_session[i]:
            signals[i] = 0.0
            continue
        
        # Skip if any required data is NaN
        if (np.isnan(jaw[i]) or np.isnan(teeth[i]) or np.isnan(lips[i]) or
            np.isnan(ema_50_1d_aligned[i]) or np.isnan(vol_sma_20[i])):
            signals[i] = 0.0
            continue
        
        # Volume filter: current volume > 2.0x 20-period volume SMA
        vol_confirm = volume[i] > (vol_sma_20[i] * 2.0)
        
        # === LONG CONDITIONS ===
        # 1. Alligator bullish alignment: jaw < teeth < lips
        # 2. Price > lips (above the Alligator's mouth)
        # 3. 1d EMA50 uptrend (close > EMA50)
        # 4. Volume confirmation
        if (jaw[i] < teeth[i]) and (teeth[i] < lips[i]) and \
           (close[i] > lips[i]) and \
           (close[i] > ema_50_1d_aligned[i]) and vol_confirm:
            signals[i] = 0.25
        
        # === SHORT CONDITIONS ===
        # 1. Alligator bearish alignment: jaw > teeth > lips
        # 2. Price < lips (below the Alligator's mouth)
        # 3. 1d EMA50 downtrend (close < EMA50)
        # 4. Volume confirmation
        elif (jaw[i] > teeth[i]) and (teeth[i] > lips[i]) and \
             (close[i] < lips[i]) and \
             (close[i] < ema_50_1d_aligned[i]) and vol_confirm:
            signals[i] = -0.25
        
        else:
            signals[i] = 0.0  # flat
    
    return signals

name = "12h_WilliamsAlligator_1dEMA50_Volume_Filter_v1"
timeframe = "12h"
leverage = 1.0
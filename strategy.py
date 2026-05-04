#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h Williams Alligator with 1w EMA trend filter and volume confirmation
# Williams Alligator uses three SMAs (Jaw=13, Teeth=8, Lips=5) to identify trends.
# Long when Lips > Teeth > Jaw (bullish alignment) AND price > Jaw AND 1w EMA34 > EMA55 (uptrend) AND volume > 1.5x 20 EMA
# Short when Lips < Teeth < Jaw (bearish alignment) AND price < Jaw AND 1w EMA34 < EMA55 (downtrend) AND volume > 1.5x 20 EMA
# Uses 12h timeframe for lower frequency, Alligator for trend alignment, 1w EMA to filter higher timeframe trend,
# volume confirmation to avoid false signals. Designed for 12-37 trades/year with discrete sizing (0.25).
# Works in bull markets via longs in strong uptrends and bear markets via shorts in strong downtrends.

name = "12h_WilliamsAlligator_1wEMA_VolumeConfirm"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Get 1w data for HTF EMA trend filter - ONCE before loop
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 55:
        return np.zeros(n)
    
    close_1w = df_1w['close'].values
    
    # Calculate 1w EMA34 and EMA55
    ema_34_1w = pd.Series(close_1w).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_55_1w = pd.Series(close_1w).ewm(span=55, adjust=False, min_periods=55).mean().values
    ema_trend_up = ema_34_1w > ema_55_1w  # Uptrend filter
    ema_trend_down = ema_34_1w < ema_55_1w  # Downtrend filter
    
    # Align 1w EMA trend to 12h timeframe
    ema_trend_up_aligned = align_htf_to_ltf(prices, df_1w, ema_trend_up.astype(float))
    ema_trend_down_aligned = align_htf_to_ltf(prices, df_1w, ema_trend_down.astype(float))
    
    # Calculate 12h SMAs for Williams Alligator: Jaw(13), Teeth(8), Lips(5)
    sma_5 = pd.Series(close).rolling(window=5, min_periods=5).mean().values
    sma_8 = pd.Series(close).rolling(window=8, min_periods=8).mean().values
    sma_13 = pd.Series(close).rolling(window=13, min_periods=13).mean().values
    
    # Williams Alligator conditions
    bullish_alignment = (sma_5 > sma_8) & (sma_8 > sma_13)  # Lips > Teeth > Jaw
    bearish_alignment = (sma_5 < sma_8) & (sma_8 < sma_13)  # Lips < Teeth < Jaw
    price_above_jaw = close > sma_13
    price_below_jaw = close < sma_13
    
    # Volume spike filter (20-period volume EMA)
    vol_ema_20 = pd.Series(volume).ewm(span=20, adjust=False, min_periods=20).mean().values
    volume_spike = volume > (vol_ema_20 * 1.5)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(100, n):
        # Skip if any value is NaN
        if (np.isnan(ema_trend_up_aligned[i]) or np.isnan(ema_trend_down_aligned[i]) or 
            np.isnan(sma_5[i]) or np.isnan(sma_8[i]) or np.isnan(sma_13[i]) or 
            np.isnan(volume_spike[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long conditions: Bullish alignment AND price above jaw AND 1w uptrend AND volume spike
            if (bullish_alignment[i] and 
                price_above_jaw[i] and 
                ema_trend_up_aligned[i] > 0.5 and 
                volume_spike[i]):
                signals[i] = 0.25
                position = 1
            # Short conditions: Bearish alignment AND price below jaw AND 1w downtrend AND volume spike
            elif (bearish_alignment[i] and 
                  price_below_jaw[i] and 
                  ema_trend_down_aligned[i] > 0.5 and 
                  volume_spike[i]):
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: Bearish alignment OR price below jaw OR 1w trend weakens
            if (bearish_alignment[i] or 
                not price_above_jaw[i] or 
                ema_trend_up_aligned[i] < 0.5):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: Bullish alignment OR price above jaw OR 1w trend weakens
            if (bullish_alignment[i] or 
                not price_below_jaw[i] or 
                ema_trend_down_aligned[i] < 0.5):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals
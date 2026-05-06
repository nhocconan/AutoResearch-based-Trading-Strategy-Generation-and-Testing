#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h Williams Alligator breakout with 1d EMA34 trend filter and volume confirmation
# Uses Williams Alligator (Jaw=13, Teeth=8, Lips=5) for trend identification on 12h
# 1d EMA34 filters for higher timeframe trend alignment to reduce whipsaw
# Volume spike (>1.8x 30-bar average) confirms breakout strength
# ATR-based trailing stop via signal=0 when price retraces 50% of ATR from extreme
# Discrete sizing 0.25 to balance return and fee drag; target 80-120 total trades over 4 years
# Williams Alligator works in both bull/bear markets by identifying strong trends
# Proven pattern: Alligator + volume + higher TF trend works on BTC/ETH in all regimes

name = "12h_WilliamsAlligator_1dEMA34_VolumeConfirm_v1"
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
    
    # Calculate HTF data ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    
    if len(df_1d) < 50:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    volume_1d = df_1d['volume'].values
    
    # Calculate 1d EMA34 trend filter
    close_1d_series = pd.Series(close_1d)
    ema34_1d = close_1d_series.ewm(span=34, adjust=False, min_periods=34).mean().values
    
    # Calculate Williams Alligator on 12h timeframe
    # Jaw: 13-period SMMA, Teeth: 8-period SMMA, Lips: 5-period SMMA
    # Using SMA as approximation for SMMA (smoothed moving average)
    high_series = pd.Series(high)
    low_series = pd.Series(low)
    close_series = pd.Series(close)
    
    # Typical price for Alligator calculation
    typical_price = (high + low + close) / 3
    typical_price_series = pd.Series(typical_price)
    
    jaw = typical_price_series.rolling(window=13, min_periods=13).mean().values
    teeth = typical_price_series.rolling(window=8, min_periods=8).mean().values
    lips = typical_price_series.rolling(window=5, min_periods=5).mean().values
    
    # Calculate volume spike filter (>1.8x 30-bar average)
    vol_ma_30 = pd.Series(volume).rolling(window=30, min_periods=30).mean().values
    volume_filter = volume > (1.8 * vol_ma_30)
    
    # Calculate ATR(14) for stoploss
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]  # First period
    atr = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    # Align HTF indicators to 12h timeframe
    ema34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema34_1d)
    jaw_aligned = align_htf_to_ltf(prices, df_1d, jaw)
    teeth_aligned = align_htf_to_ltf(prices, df_1d, teeth)
    lips_aligned = align_htf_to_ltf(prices, df_1d, lips)
    volume_filter_aligned = align_htf_to_ltf(prices, df_1d, volume_filter)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    long_extreme = 0.0
    short_extreme = 0.0
    
    for i in range(100, n):
        # Skip if any critical value is NaN
        if (np.isnan(ema34_1d_aligned[i]) or np.isnan(jaw_aligned[i]) or 
            np.isnan(teeth_aligned[i]) or np.isnan(lips_aligned[i]) or 
            np.isnan(volume_filter_aligned[i]) or np.isnan(atr[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long entry: Lips > Teeth > Jaw (bullish alignment) AND price > Lips AND uptrend (price > EMA34) AND volume spike
            if lips_aligned[i] > teeth_aligned[i] and teeth_aligned[i] > jaw_aligned[i] and \
               close[i] > lips_aligned[i] and close[i] > ema34_1d_aligned[i] and volume_filter_aligned[i]:
                signals[i] = 0.25
                position = 1
                long_extreme = close[i]
            # Short entry: Lips < Teeth < Jaw (bearish alignment) AND price < Lips AND downtrend (price < EMA34) AND volume spike
            elif lips_aligned[i] < teeth_aligned[i] and teeth_aligned[i] < jaw_aligned[i] and \
                 close[i] < lips_aligned[i] and close[i] < ema34_1d_aligned[i] and volume_filter_aligned[i]:
                signals[i] = -0.25
                position = -1
                short_extreme = close[i]
        elif position == 1:
            # Update long extreme
            if close[i] > long_extreme:
                long_extreme = close[i]
            # Exit long: price retraces 50% of ATR from extreme OR Alligator reverses
            if close[i] <= long_extreme - 0.5 * atr[i] or \
               lips_aligned[i] < teeth_aligned[i]:  # Alligator turning bearish
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Update short extreme
            if close[i] < short_extreme:
                short_extreme = close[i]
            # Exit short: price retraces 50% of ATR from extreme OR Alligator reverses
            if close[i] >= short_extreme + 0.5 * atr[i] or \
               lips_aligned[i] > teeth_aligned[i]:  # Alligator turning bullish
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals
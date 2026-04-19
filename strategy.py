#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "6h_1d_Alligator_Adx_Momentum"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1d data for Williams Alligator and ADX (once before loop)
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Williams Alligator: SMA of median price
    median_price_1d = (high_1d + low_1d) / 2.0
    jaw = pd.Series(median_price_1d).rolling(window=13, min_periods=13).mean().shift(8).values
    teeth = pd.Series(median_price_1d).rolling(window=8, min_periods=8).mean().shift(5).values
    lips = pd.Series(median_price_1d).rolling(window=5, min_periods=5).mean().shift(3).values
    
    # ADX: Trend strength indicator
    # +DM, -DM, TR calculation
    up_move = high_1d[1:] - high_1d[:-1]
    down_move = low_1d[:-1] - low_1d[1:]
    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0.0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0.0)
    tr = np.maximum(
        high_1d[1:] - low_1d[1:],
        np.maximum(
            np.abs(high_1d[1:] - close_1d[:-1]),
            np.abs(low_1d[1:] - close_1d[:-1])
        )
    )
    
    # Smooth with Wilder's smoothing (EMA with alpha=1/period)
    def wilders_smoothing(data, period):
        alpha = 1.0 / period
        result = np.zeros_like(data)
        result[0] = data[0] if len(data) > 0 else 0
        for i in range(1, len(data)):
            result[i] = alpha * data[i] + (1 - alpha) * result[i-1]
        return result
    
    period = 14
    atr = wilders_smoothing(tr, period)
    plus_di = 100 * wilders_smoothing(plus_dm, period) / atr
    minus_di = 100 * wilders_smoothing(minus_dm, period) / atr
    dx = 100 * np.abs(plus_di - minus_di) / (plus_di + minus_di)
    adx = wilders_smoothing(dx, period)
    
    # Align indicators to 6h timeframe
    jaw_6h = align_htf_to_ltf(prices, df_1d, jaw)
    teeth_6h = align_htf_to_ltf(prices, df_1d, teeth)
    lips_6h = align_htf_to_ltf(prices, df_1d, lips)
    adx_6h = align_htf_to_ltf(prices, df_1d, adx)
    
    # Momentum: 6-period ROC on 6h close
    roc = np.zeros_like(close)
    roc[6:] = (close[6:] - close[:-6]) / close[:-6] * 100
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 50
    
    for i in range(start_idx, n):
        if np.isnan(jaw_6h[i]) or np.isnan(teeth_6h[i]) or np.isnan(lips_6h[i]) or \
           np.isnan(adx_6h[i]) or np.isnan(roc[i]):
            signals[i] = 0.0
            continue
        
        price = close[i]
        
        # Alligator alignment: lips > teeth > jaw = bullish, lips < teeth < jaw = bearish
        bullish_align = lips_6h[i] > teeth_6h[i] and teeth_6h[i] > jaw_6h[i]
        bearish_align = lips_6h[i] < teeth_6h[i] and teeth_6h[i] < jaw_6h[i]
        
        # Strong trend filter
        strong_trend = adx_6h[i] > 25
        
        # Momentum confirmation
        mom_bullish = roc[i] > 0
        mom_bearish = roc[i] < 0
        
        if position == 0:
            # Long: Bullish alignment + strong trend + positive momentum
            if bullish_align and strong_trend and mom_bullish:
                signals[i] = 0.25
                position = 1
            # Short: Bearish alignment + strong trend + negative momentum
            elif bearish_align and strong_trend and mom_bearish:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit: Bearish alignment or weak trend
            if not bullish_align or adx_6h[i] < 20:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit: Bullish alignment or weak trend
            if not bearish_align or adx_6h[i] < 20:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals
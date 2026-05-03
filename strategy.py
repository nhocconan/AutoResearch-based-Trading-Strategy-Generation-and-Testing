#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1d Williams Alligator with 1w EMA50 trend filter and volume confirmation.
# Long when price > Alligator Jaw (13-period SMMA) in bull trend (close > 1w EMA50) with volume > 2.0x 20-period MA.
# Short when price < Alligator Jaw in bear trend (close < 1w EMA50) with volume spike.
# Uses discrete position sizing (0.25) to balance return and drawdown. Williams Alligator provides smoothed trend structure,
# ideal for 1d timeframe. Volume confirmation ensures institutional participation. 1w trend filter reduces whipsaw vs shorter MAs.
# Target: 30-100 total trades over 4 years (7-25/year).

name = "1d_WilliamsAlligator_1wEMA50_Volume"
timeframe = "1d"
leverage = 1.0

def smma(data, period):
    """Smoothed Moving Average (SMMA) - also called Wilder's Moving Average"""
    if len(data) < period:
        return np.full_like(data, np.nan, dtype=float)
    result = np.full_like(data, np.nan, dtype=float)
    # First value is simple moving average
    result[period-1] = np.mean(data[:period])
    # Subsequent values: SMMA = (prev_SMMA * (period-1) + current_price) / period
    for i in range(period, len(data)):
        result[i] = (result[i-1] * (period-1) + data[i]) / period
    return result

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Get 1w data for EMA trend filter
    df_1w = get_htf_data(prices, '1w')
    
    if len(df_1w) < 2:
        return np.zeros(n)
    
    # Calculate 1w EMA50 for trend filter
    ema_50_1w = pd.Series(df_1w['close'].values).ewm(span=50, min_periods=50, adjust=False).mean().values
    
    # Align EMA to 1d timeframe
    ema_50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_50_1w)
    
    # Williams Alligator components (13, 8, 5 period SMMA of median price)
    median_price = (high + low) / 2
    jaw = smma(median_price, 13)  # Jaw (13-period SMMA)
    teeth = smma(median_price, 8)   # Teeth (8-period SMMA)
    lips = smma(median_price, 5)    # Lips (5-period SMMA)
    
    # Volume regime: current 1d volume > 2.0x 20-period MA
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (2.0 * vol_ma_20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(100, n):
        # Skip if any value is NaN
        if (np.isnan(ema_50_1w_aligned[i]) or np.isnan(jaw[i]) or 
            np.isnan(teeth[i]) or np.isnan(lips[i]) or np.isnan(vol_ma_20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
            
        close_val = close[i]
        ema_trend = ema_50_1w_aligned[i]
        jaw_val = jaw[i]
        vol_spike = volume_spike[i]
        
        # Determine trend regime
        is_bull_trend = close_val > ema_trend
        is_bear_trend = close_val < ema_trend
        
        # Alligator condition: price relative to Jaw
        price_above_jaw = close_val > jaw_val
        price_below_jaw = close_val < jaw_val
        
        # Entry logic
        if position == 0:
            if is_bull_trend and price_above_jaw and vol_spike:
                signals[i] = 0.25
                position = 1
            elif is_bear_trend and price_below_jaw and vol_spike:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Long exit: price below Jaw OR trend reversal
            if price_below_jaw or close_val < ema_trend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: price above Jaw OR trend reversal
            if price_above_jaw or close_val > ema_trend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals
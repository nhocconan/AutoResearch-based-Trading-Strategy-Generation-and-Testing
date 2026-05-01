#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1d Williams Alligator with 1w EMA50 trend filter and volume confirmation.
# Long when price > Alligator Jaw AND Alligator Teeth > Alligator Lips (bullish alignment) AND price > 1w EMA50 AND volume > 1.5x 20-bar average.
# Short when price < Alligator Jaw AND Alligator Teeth < Alligator Lips (bearish alignment) AND price < 1w EMA50 AND volume > 1.5x 20-bar average.
# Uses discrete sizing 0.25 to manage drawdown. Session filter 08-20 UTC to avoid low-liquidity hours.
# Williams Alligator uses smoothed moving averages (SMMA) of median price with periods 13,8,5 and offsets 8,5,3.
# This strategy targets 30-80 total trades over 4 years (7-20/year) for 1d timeframe.
# 1w EMA50 provides robust multi-week trend alignment that works in both bull (price above EMA) and bear (price below EMA).
# Williams Alligator provides clear trend alignment signals with built-in smoothing to reduce noise.
# Volume confirmation (1.5x average) ensures only high-conviction moves are traded, reducing overtrading.

name = "1d_WilliamsAlligator_1wEMA50_Trend_VolumeConfirm_v1"
timeframe = "1d"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    open_time = prices['open_time']
    
    # Pre-compute session hours for efficiency (08-20 UTC)
    hours = pd.DatetimeIndex(open_time).hour
    
    # Load 1w data ONCE before loop for EMA50 trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 50:
        return np.zeros(n)
    
    # 1w EMA50 calculation
    close_1w = df_1w['close'].values
    ema_50 = pd.Series(close_1w).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_aligned = align_htf_to_ltf(prices, df_1w, ema_50)
    
    # 1w trend: price above/below EMA50
    price_above_ema = close > ema_50_aligned
    price_below_ema = close < ema_50_aligned
    
    # Williams Alligator: Smoothed Moving Average (SMMA) of Median Price
    # Median Price = (high + low) / 2
    median_price = (high + low) / 2.0
    
    # SMMA calculation (similar to Wilder's smoothing, alpha = 1/period)
    def smma(data, period):
        result = np.full_like(data, np.nan)
        if len(data) < period:
            return result
        # First value is simple average
        result[period-1] = np.mean(data[:period])
        # Subsequent values: SMMA = (Prev SMMA * (period-1) + Current Price) / period
        for i in range(period, len(data)):
            result[i] = (result[i-1] * (period-1) + data[i]) / period
        return result
    
    # Alligator Lines: Jaw (13,8), Teeth (8,5), Lips (5,3)
    jaw = smma(median_price, 13)
    teeth = smma(median_price, 8)
    lips = smma(median_price, 5)
    
    # Apply offsets: Jaw offset 8, Teeth offset 5, Lips offset 3
    jaw = np.roll(jaw, 8)
    teeth = np.roll(teeth, 5)
    lips = np.roll(lips, 3)
    
    # Alligator alignment conditions
    bullish_alignment = (teeth > lips) & (jaw > teeth)  # Teeth > Lips AND Jaw > Teeth
    bearish_alignment = (teeth < lips) & (jaw < teeth)  # Teeth < Lips AND Jaw < Teeth
    
    # Volume confirmation: current 1d volume > 1.5x 20-bar average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 50  # warmup for EMA, volume MA, and Alligator (max offset 8 + max period 13 = 21, but use 50 for safety)
    
    for i in range(start_idx, n):
        # Session filter: trade only 08-20 UTC
        hour = hours[i]
        if hour < 8 or hour > 20:
            signals[i] = 0.0
            continue
        
        if np.isnan(ema_50_aligned[i]) or np.isnan(jaw[i]) or np.isnan(teeth[i]) or np.isnan(lips[i]) or np.isnan(vol_ma[i]):
            signals[i] = 0.0
            continue
        
        curr_close = close[i]
        curr_vol = volume[i]
        curr_vol_ma = vol_ma[i]
        
        if curr_vol_ma <= 0:
            signals[i] = 0.0
            continue
            
        volume_confirm = curr_vol > (curr_vol_ma * 1.5)
        
        # Entry conditions
        if position == 0:  # Flat - look for new entries
            # Long: price > Jaw AND bullish alignment AND price > 1w EMA50 AND volume confirmation
            if (curr_close > jaw[i] and 
                bullish_alignment[i] and 
                price_above_ema[i] and 
                volume_confirm):
                signals[i] = 0.25
                position = 1
            # Short: price < Jaw AND bearish alignment AND price < 1w EMA50 AND volume confirmation
            elif (curr_close < jaw[i] and 
                  bearish_alignment[i] and 
                  price_below_ema[i] and 
                  volume_confirm):
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        
        elif position == 1:  # Long position
            # Exit: price crosses below Jaw OR bearish alignment OR price < 1w EMA50 (trend change)
            if (curr_close < jaw[i] or 
                bearish_alignment[i] or 
                not price_above_ema[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:  # Short position
            # Exit: price crosses above Jaw OR bullish alignment OR price > 1w EMA50 (trend change)
            if (curr_close > jaw[i] or 
                bullish_alignment[i] or 
                not price_below_ema[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals
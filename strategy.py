#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1d Camarilla R3/S3 breakout with 1w EMA34 trend filter and volume spike confirmation
# Long when price closes above Camarilla R3 level AND 1w trend is bullish (close > EMA34) AND volume > 2.0x 20-period volume EMA
# Short when price closes below Camarilla S3 level AND 1w trend is bearish (close < EMA34) AND volume > 2.0x 20-period volume EMA
# Exit when price reverses to Camarilla pivot (PP) level OR 1w trend changes direction
# Uses discrete position size 0.30 to balance return and drawdown. Volume spike at 2.0x reduces overtrading.
# Camarilla levels provide intraday-relevant support/resistance that works well on daily timeframe for breakouts.
# 1w EMA34 filters major trend to avoid counter-trend whipsaws in ranging markets.

name = "1d_Camarilla_R3S3_1wTrend_VolumeSpike"
timeframe = "1d"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Get 1w data for HTF trend filter - ONCE before loop
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 34:
        return np.zeros(n)
    
    close_1w = df_1w['close'].values
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    
    # Calculate 1w EMA34 for trend filter
    ema_34_1w = pd.Series(close_1w).ewm(span=34, adjust=False, min_periods=34).mean().values
    trend_bullish_1w = close_1w > ema_34_1w
    trend_bearish_1w = close_1w < ema_34_1w
    
    # Align 1w trend to 1d timeframe
    trend_bullish_aligned = align_htf_to_ltf(prices, df_1w, trend_bullish_1w.astype(float))
    trend_bearish_aligned = align_htf_to_ltf(prices, df_1w, trend_bearish_1w.astype(float))
    
    # Calculate Camarilla levels from previous day's OHLC
    # Camarilla: PP = (H + L + C) / 3
    # R3 = PP + (H - L) * 1.1/2
    # S3 = PP - (H - L) * 1.1/2
    prev_high = np.roll(high, 1)
    prev_low = np.roll(low, 1)
    prev_close = np.roll(close, 1)
    
    # Handle first bar
    prev_high[0] = high[0]
    prev_low[0] = low[0]
    prev_close[0] = close[0]
    
    PP = (prev_high + prev_low + prev_close) / 3.0
    R3 = PP + (prev_high - prev_low) * 1.1 / 2.0
    S3 = PP - (prev_high - prev_low) * 1.1 / 2.0
    
    # Calculate volume spike filter (20-period volume EMA)
    vol_ema_20 = pd.Series(volume).ewm(span=20, adjust=False, min_periods=20).mean().values
    volume_spike = volume > (vol_ema_20 * 2.0)  # Volume at least 2.0x average for confirmation
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(100, n):
        # Skip if any value is NaN
        if (np.isnan(trend_bullish_aligned[i]) or np.isnan(trend_bearish_aligned[i]) or 
            np.isnan(R3[i]) or np.isnan(S3[i]) or 
            np.isnan(volume_spike[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long conditions: price closes above Camarilla R3 AND 1w bullish trend AND volume spike
            if (close[i] > R3[i] and 
                trend_bullish_aligned[i] > 0.5 and  # 1w bullish trend
                volume_spike[i]):
                signals[i] = 0.30
                position = 1
            # Short conditions: price closes below Camarilla S3 AND 1w bearish trend AND volume spike
            elif (close[i] < S3[i] and 
                  trend_bearish_aligned[i] > 0.5 and  # 1w bearish trend
                  volume_spike[i]):
                signals[i] = -0.30
                position = -1
        elif position == 1:
            # Exit long: price closes below Camarilla pivot (PP) OR 1w trend turns bearish
            if (close[i] < PP[i] or 
                trend_bearish_aligned[i] > 0.5):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.30
        elif position == -1:
            # Exit short: price closes above Camarilla pivot (PP) OR 1w trend turns bullish
            if (close[i] > PP[i] or 
                trend_bullish_aligned[i] > 0.5):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.30
    
    return signals
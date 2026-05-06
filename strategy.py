#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1d strategy using 1w Volatility Contraction Pattern (VCP) and 1d Institutional Buying Pressure
# - Uses 1w ATR contraction (ATR < 50th percentile) to identify low volatility accumulation phases
# - Uses 1d OBV slope (20-period linear regression slope) to detect institutional accumulation
# - Enters long when price breaks above 1d high with expanding volume in accumulation phase
# - Exits when price closes below 1d VWAP or volatility expands (ATR > 80th percentile)
# - Designed to capture institutional accumulation during weekly contractions with daily confirmation
# - Target: 40-80 total trades over 4 years (10-20/year) with 0.25 position sizing

name = "1d_1wVCP_1dOBV_Accumulation"
timeframe = "1d"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1d data for price action and OBV
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # Get 1w data for volatility contraction measurement
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 50:
        return np.zeros(n)
    
    # Calculate 1d high and low for breakout levels
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate 1d VWAP
    typical_price = (high_1d + low_1d + close_1d) / 3
    vwap_numerator = np.cumsum(typical_price * volume)
    vwap_denominator = np.cumsum(volume)
    vwap = np.where(vwap_denominator != 0, vwap_numerator / vwap_denominator, 0)
    
    # Calculate 1d OBV (On-Balance Volume)
    obv = np.zeros_like(close_1d)
    obv[0] = volume[0]
    for i in range(1, len(close_1d)):
        if close_1d[i] > close_1d[i-1]:
            obv[i] = obv[i-1] + volume[i]
        elif close_1d[i] < close_1d[i-1]:
            obv[i] = obv[i-1] - volume[i]
        else:
            obv[i] = obv[i-1]
    
    # Calculate 1d OBV 20-period linear regression slope (institutional buying pressure)
    def linreg_slope(values, period):
        if len(values) < period:
            return np.full_like(values, np.nan)
        result = np.full(len(values), np.nan)
        for i in range(period-1, len(values)):
            y = values[i-period+1:i+1]
            x = np.arange(period)
            slope = np.polyfit(x, y, 1)[0]
            result[i] = slope
        return result
    
    obv_slope = linreg_slope(obv, 20)
    
    # Calculate 1w ATR for volatility contraction measurement
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    
    # True Range
    tr1 = high_1w - low_1w
    tr2 = np.abs(high_1w - np.roll(close_1w, 1))
    tr3 = np.abs(low_1w - np.roll(close_1w, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]  # First period
    
    # ATR using Wilder's smoothing (14-period)
    def wilders_smoothing(data, period):
        result = np.zeros_like(data)
        if len(data) < period:
            return result
        result[period-1] = np.nansum(data[:period])
        for i in range(period, len(data)):
            result[i] = result[i-1] - (result[i-1] / period) + data[i]
        return result
    
    atr_14 = wilders_smoothing(tr, 14)
    
    # Calculate 1w ATR percentile rank (lookback 50 periods)
    atr_series = pd.Series(atr_14)
    atr_percentile = atr_series.rolling(window=50, min_periods=30).apply(
        lambda x: pd.Series(x).rank(pct=True).iloc[-1] * 100, raw=False
    ).values
    
    # Align 1d indicators to 1d timeframe (no alignment needed for same timeframe)
    # Align 1w ATR percentile to 1d timeframe
    atr_percentile_1d = align_htf_to_ltf(prices, df_1w, atr_percentile)
    
    # Volume confirmation (expanding volume on breakout)
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_expanding = volume > vol_ma_20  # Volume above average
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long
    
    for i in range(100, n):  # Start after warmup
        # Skip if any critical value is NaN
        if (np.isnan(high_1d[i]) or np.isnan(low_1d[i]) or np.isnan(close_1d[i]) or
            np.isnan(vwap[i]) or np.isnan(obv_slope[i]) or np.isnan(atr_percentile_1d[i]) or
            np.isnan(volume_expanding[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Look for volatility contraction (ATR < 50th percentile) and institutional buying (OBV slope > 0)
            volatility_contraction = atr_percentile_1d[i] < 50
            institutional_buying = obv_slope[i] > 0
            
            if volatility_contraction and institutional_buying:
                # Long: price breaks above 1d high with expanding volume
                if close[i] > high_1d[i] and volume_expanding[i]:
                    signals[i] = 0.25
                    position = 1
        elif position == 1:
            # Exit long: price closes below 1d VWAP OR volatility expands (ATR > 80th percentile)
            if close[i] < vwap[i] or atr_percentile_1d[i] > 80:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
    
    return signals
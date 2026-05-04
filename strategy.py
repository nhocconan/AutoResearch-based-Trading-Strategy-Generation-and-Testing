#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h Williams Alligator + 1w ADX regime + volume confirmation
# Williams Alligator (Jaw=13, Teeth=8, Lips=5) identifies trend: 
# - Jaw > Teeth > Lips = uptrend (long bias)
# - Jaw < Teeth < Lips = downtrend (short bias)
# - Intertwined = ranging (no trade)
# In trending markets (1w ADX>=25), we trade Alligator alignment: long when bullish, short when bearish.
# Volume confirmation (>1.3x 20-period EMA) filters weak signals.
# Designed for 12h timeframe targeting 50-150 total trades over 4 years.
# Uses discrete position sizing (0.25) to minimize fee churn and manage drawdown.

name = "12h_WilliamsAlligator_1wADX_Regime_Volume"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1w data for ADX
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 30:
        return np.zeros(n)
    
    # Calculate 1w ADX (14-period)
    plus_dm = pd.Series(df_1w['high']).diff()
    minus_dm = pd.Series(df_1w['low']).diff().mul(-1)
    plus_dm = plus_dm.where((plus_dm > minus_dm) & (plus_dm > 0), 0.0)
    minus_dm = minus_dm.where((minus_dm > plus_dm) & (minus_dm > 0), 0.0)
    tr1 = pd.Series(df_1w['high']).sub(df_1w['low'])
    tr2 = pd.Series(df_1w['high']).sub(df_1w['close'].shift(1)).abs()
    tr3 = pd.Series(df_1w['low']).sub(df_1w['close'].shift(1)).abs()
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr = tr.rolling(window=14, min_periods=14).mean()
    plus_di = 100 * (plus_dm.rolling(window=14, min_periods=14).sum() / atr)
    minus_di = 100 * (minus_dm.rolling(window=14, min_periods=14).sum() / atr)
    dx = (abs(plus_di - minus_di) / (plus_di + minus_di)) * 100
    adx = dx.rolling(window=14, min_periods=14).mean()
    
    # Align 1w ADX to 12h timeframe
    adx_aligned = align_htf_to_ltf(prices, df_1w, adx.values)
    
    # Williams Alligator on 12h timeframe
    # Jaw: 13-period SMMA, Teeth: 8-period SMMA, Lips: 5-period SMMA
    # SMMA = Smoothed Moving Average (similar to Wilder's smoothing)
    def smma(arr, period):
        result = np.full_like(arr, np.nan, dtype=np.float64)
        if len(arr) < period:
            return result
        # First value is simple SMA
        result[period-1] = np.mean(arr[:period])
        # Subsequent values: SMMA = (prev_SMMA * (period-1) + current_price) / period
        for i in range(period, len(arr)):
            result[i] = (result[i-1] * (period-1) + arr[i]) / period
        return result
    
    jaw = smma(high, 13)  # Typically applied to median price, but high works for trend
    teeth = smma(high, 8)
    lips = smma(high, 5)
    
    # Volume confirmation: 20-period EMA of volume on 12h timeframe
    vol_ema_20 = pd.Series(volume).ewm(span=20, adjust=False, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(100, n):
        # Skip if any value is NaN
        if (np.isnan(adx_aligned[i]) or np.isnan(jaw[i]) or np.isnan(teeth[i]) or 
            np.isnan(lips[i]) or np.isnan(vol_ema_20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Volume confirmation: current volume > 1.3 x 20-period EMA
        volume_confirm = volume[i] > (1.3 * vol_ema_20[i])
        
        if position == 0:
            # Only trade in trending markets (1w ADX>=25)
            if adx_aligned[i] >= 25 and volume_confirm:
                # Bullish alignment: Jaw > Teeth > Lips
                if jaw[i] > teeth[i] and teeth[i] > lips[i]:
                    signals[i] = 0.25
                    position = 1
                # Bearish alignment: Jaw < Teeth < Lips
                elif jaw[i] < teeth[i] and teeth[i] < lips[i]:
                    signals[i] = -0.25
                    position = -1
        elif position == 1:
            # Exit long: Alligator alignment breaks (teeth crosses below jaw) OR ADX weakens (<20) OR volume drops
            if (teeth[i] < jaw[i] or 
                adx_aligned[i] < 20 or 
                volume[i] < vol_ema_20[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: Alligator alignment breaks (teeth crosses above jaw) OR ADX weakens (<20) OR volume drops
            if (teeth[i] > jaw[i] or 
                adx_aligned[i] < 20 or 
                volume[i] < vol_ema_20[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals
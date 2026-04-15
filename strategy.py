#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Williams Alligator + Elder Ray Power with 1d regime filter
# Uses Alligator (JAWS/TEETH/LIPS) to identify trendless markets and Elder Ray to measure bull/bear power
# Long when: Elder Bull Power > 0 AND price above Alligator TEETH AND 1d ADX < 25 (range regime) for mean reversion
# Short when: Elder Bear Power < 0 AND price below Alligator TEETH AND 1d ADX < 25 (range regime) for mean reversion
# Uses discrete position sizing (0.25) to control drawdown and minimize fee drag.
# Alligator identifies sideways markets where Elder Ray works best for mean reversion.
# 1d ADX < 25 filter ensures we only trade in ranging conditions on higher timeframe.

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
    if len(df_1d) < 30:
        return np.zeros(n)
    
    # === 1d Indicator: ADX for regime filter ===
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # True Range calculation
    tr1 = np.abs(high_1d[1:] - low_1d[1:])
    tr2 = np.abs(high_1d[1:] - close_1d[:-1])
    tr3 = np.abs(low_1d[1:] - close_1d[:-1])
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr = np.concatenate([[np.nan], tr])  # align with index
    
    # Directional Movement
    dm_plus = np.where((high_1d[1:] - high_1d[:-1]) > (low_1d[:-1] - low_1d[1:]), 
                       np.maximum(high_1d[1:] - high_1d[:-1], 0), 0)
    dm_minus = np.where((low_1d[:-1] - low_1d[1:]) > (high_1d[1:] - high_1d[:-1]), 
                        np.maximum(low_1d[:-1] - low_1d[1:], 0), 0)
    dm_plus = np.concatenate([[np.nan], dm_plus])
    dm_minus = np.concatenate([[np.nan], dm_minus])
    
    # Smoothed TR, DM+ and DM-
    def ma_smoother(data, period):
        return pd.Series(data).ewm(alpha=1/period, adjust=False, min_periods=period).mean().values
    
    tr_ma = ma_smoother(tr, 14)
    dm_plus_ma = ma_smoother(dm_plus, 14)
    dm_minus_ma = ma_smoother(dm_minus, 14)
    
    # DI+ and DI-
    di_plus = 100 * dm_plus_ma / tr_ma
    di_minus = 100 * dm_minus_ma / tr_ma
    
    # DX and ADX
    dx = 100 * np.abs(di_plus - di_minus) / (di_plus + di_minus)
    adx = ma_smoother(dx, 14)
    adx_1d_aligned = align_htf_to_ltf(prices, df_1d, adx)
    
    # === 6h Indicators: Williams Alligator ===
    # JAWS (Blue): 13-period SMMA smoothed 8 bars ahead
    # TEETH (Red): 8-period SMMA smoothed 5 bars ahead  
    # LIPS (Green): 5-period SMMA smoothed 3 bars ahead
    def smma(data, period):
        # Smoothed Moving Average
        sma = pd.Series(data).rolling(window=period, min_periods=period).mean().values
        smma_vals = np.full_like(data, np.nan, dtype=float)
        if len(sma) >= period:
            smma_vals[period-1] = sma[period-1]
            for i in range(period, len(data)):
                smma_vals[i] = (smma_vals[i-1] * (period-1) + data[i]) / period
        return smma_vals
    
    jaws = smma(close, 13)
    teeth = smma(close, 8)
    lips = smma(close, 5)
    
    # Shift Alligator lines forward (JAWS: +8, TEETH: +5, LIPS: +3)
    jaws_shifted = np.concatenate([np.full(8, np.nan), jaws[:-8]]) if len(jaws) > 8 else np.full_like(jaws, np.nan)
    teeth_shifted = np.concatenate([np.full(5, np.nan), teeth[:-5]]) if len(teeth) > 5 else np.full_like(teeth, np.nan)
    lips_shifted = np.concatenate([np.full(3, np.nan), lips[:-3]]) if len(lips) > 3 else np.full_like(lips, np.nan)
    
    # === 6h Indicators: Elder Ray Power ===
    # Bull Power = High - EMA(13)
    # Bear Power = Low - EMA(13)
    ema_13 = pd.Series(close).ewm(span=13, adjust=False, min_periods=13).mean().values
    bull_power = high - ema_13
    bear_power = low - ema_13
    
    signals = np.zeros(n)
    
    # Warmup: ensure all indicators are valid
    warmup = max(50, 20) + 10  # sufficient buffer for all indicators
    
    for i in range(warmup, n):
        # Skip if outside trading session (08-20 UTC)
        if not in_session[i]:
            signals[i] = 0.0
            continue
        
        # Skip if any required data is NaN
        if (np.isnan(jaws_shifted[i]) or np.isnan(teeth_shifted[i]) or np.isnan(lips_shifted[i]) or
            np.isnan(bull_power[i]) or np.isnan(bear_power[i]) or np.isnan(adx_1d_aligned[i])):
            signals[i] = 0.0
            continue
        
        # === LONG CONDITIONS ===
        # 1. Elder Bull Power > 0 (bulls in control)
        # 2. Price above Alligator TEETH (trend bias up)
        # 3. 1d ADX < 25 (range regime on higher timeframe for mean reversion)
        if (bull_power[i] > 0) and \
           (close[i] > teeth_shifted[i]) and \
           (adx_1d_aligned[i] < 25):
            signals[i] = 0.25
        
        # === SHORT CONDITIONS ===
        # 1. Elder Bear Power < 0 (bears in control)
        # 2. Price below Alligator TEETH (trend bias down)
        # 3. 1d ADX < 25 (range regime on higher timeframe for mean reversion)
        elif (bear_power[i] < 0) and \
             (close[i] < teeth_shifted[i]) and \
             (adx_1d_aligned[i] < 25):
            signals[i] = -0.25
        
        else:
            signals[i] = 0.0  # flat
    
    return signals

name = "6h_Alligator_ElderRay_1dADX_RangeFilter_v1"
timeframe = "6h"
leverage = 1.0
#!/usr/bin/env python3
# Hypothesis: 6h Elder Ray + 1d ADX regime filter + volume spike confirmation. Elder Ray measures bull/bear power via EMA13. In bull regimes (ADX>25, +DI>-DI), go long when bull power turns positive with volume spike. In bear regimes (ADX>25, -DI>+DI), go short when bear power turns negative with volume spike. In ranging regimes (ADX<20), fade extreme Elder Ray values at 2.0 std dev bands. Uses discrete sizing (0.0, ±0.25) to limit fee churn. Targets 50-150 trades over 4 years.

name = "6h_ElderRay_ADX_Regime_VolumeSpike_v1"
timeframe = "6h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    open_ = prices['open'].values
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # --- 6h Indicators (LTF) ---
    # EMA13 for Elder Ray
    ema13 = pd.Series(close).ewm(span=13, adjust=False, min_periods=13).mean().values
    # Bull Power = High - EMA13, Bear Power = EMA13 - Low
    bull_power = high - ema13
    bear_power = ema13 - low
    # Volume spike: > 2.0x 20-period average
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (2.0 * vol_ma_20)
    
    # --- 1d Indicators (HTF) ---
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # ADX(14) calculation
    plus_dm = np.diff(high_1d, prepend=high_1d[0])
    minus_dm = np.diff(low_1d, prepend=low_1d[0])
    plus_dm = np.where((plus_dm > minus_dm) & (plus_dm > 0), plus_dm, 0.0)
    minus_dm = np.where((minus_dm > plus_dm) & (minus_dm > 0), minus_dm, 0.0)
    
    tr = np.maximum(
        np.absolute(np.diff(high_1d, prepend=high_1d[0])),
        np.maximum(
            np.absolute(np.diff(low_1d, prepend=low_1d[0])),
            np.absolute(np.diff(close_1d, prepend=close_1d[0]))
        )
    )
    atr_14 = pd.Series(tr).ewm(span=14, adjust=False, min_periods=14).mean().values
    
    plus_di_14 = 100 * pd.Series(plus_dm).ewm(span=14, adjust=False, min_periods=14).mean().values / atr_14
    minus_di_14 = 100 * pd.Series(minus_dm).ewm(span=14, adjust=False, min_periods=14).mean().values / atr_14
    dx = 100 * np.absolute(plus_di_14 - minus_di_14) / (plus_di_14 + minus_di_14 + 1e-10)
    adx_14 = pd.Series(dx).ewm(span=14, adjust=False, min_periods=14).mean().values
    
    # Align HTF indicators to LTF
    adx_14_aligned = align_htf_to_ltf(prices, df_1d, adx_14)
    plus_di_14_aligned = align_htf_to_ltf(prices, df_1d, plus_di_14)
    minus_di_14_aligned = align_htf_to_ltf(prices, df_1d, minus_di_14)
    
    # Elder Ray std bands for ranging regime (20-period)
    bull_power_ma = pd.Series(bull_power).rolling(window=20, min_periods=20).mean().values
    bull_power_std = pd.Series(bull_power).rolling(window=20, min_periods=20).std().values
    bear_power_ma = pd.Series(bear_power).rolling(window=20, min_periods=20).mean().values
    bear_power_std = pd.Series(bear_power).rolling(window=20, min_periods=20).std().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(1, n):
        # Skip if missing data
        if (np.isnan(ema13[i]) or np.isnan(bull_power[i]) or np.isnan(bear_power[i]) or
            np.isnan(volume_spike[i]) or np.isnan(adx_14_aligned[i]) or
            np.isnan(plus_di_14_aligned[i]) or np.isnan(minus_di_14_aligned[i]) or
            np.isnan(bull_power_ma[i]) or np.isnan(bull_power_std[i]) or
            np.isnan(bear_power_ma[i]) or np.isnan(bear_power_std[i])):
            signals[i] = 0.0
            continue
        
        adx = adx_14_aligned[i]
        plus_di = plus_di_14_aligned[i]
        minus_di = minus_di_14_aligned[i]
        
        if position == 0:
            # Trending regime: ADX > 25
            if adx > 25:
                # Bullish trend: +DI > -DI
                if plus_di > minus_di:
                    # Long when bull power turns positive with volume spike
                    if bull_power[i] > 0 and bull_power[i-1] <= 0 and volume_spike[i]:
                        signals[i] = 0.25
                        position = 1
                # Bearish trend: -DI > +DI
                elif minus_di > plus_di:
                    # Short when bear power turns negative with volume spike
                    if bear_power[i] > 0 and bear_power[i-1] <= 0 and volume_spike[i]:
                        signals[i] = -0.25
                        position = -1
            # Ranging regime: ADX < 20
            elif adx < 20:
                # Long when bull power crosses above +2 std dev band (oversold bounce)
                if bull_power[i] > (bull_power_ma[i] + 2.0 * bull_power_std[i]) and \
                   bull_power[i-1] <= (bull_power_ma[i-1] + 2.0 * bull_power_std[i-1]):
                    signals[i] = 0.25
                    position = 1
                # Short when bear power crosses above +2 std dev band (overbought fade)
                elif bear_power[i] > (bear_power_ma[i] + 2.0 * bear_power_std[i]) and \
                     bear_power[i-1] <= (bear_power_ma[i-1] + 2.0 * bear_power_std[i-1]):
                    signals[i] = -0.25
                    position = -1
        elif position == 1:
            # Exit long: trend change or mean reversion signal
            if adx > 25 and minus_di > plus_di:  # trend turned bearish
                signals[i] = 0.0
                position = 0
            elif adx < 20 and bull_power[i] < bull_power_ma[i]:  # mean reversion in range
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: trend change or mean reversion signal
            if adx > 25 and plus_di > minus_di:  # trend turned bullish
                signals[i] = 0.0
                position = 0
            elif adx < 20 and bear_power[i] < bear_power_ma[i]:  # mean reversion in range
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals
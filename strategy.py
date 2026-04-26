#!/usr/bin/env python3
"""
4h_Camarilla_R3S3_Breakout_1dEMA34_Trend_Volume_RegimeFilter
Hypothesis: On 4h timeframe, Camarilla R3/S3 breakouts with 1d EMA34 trend filter, volume spike (>2.0x 20-bar avg), and regime filter (ADX < 25 for ranging, ADX > 25 for trending) captures high-probability institutional breakouts. ADX regime filter adapts: in ranging markets (ADX<25) we fade breaks at R3/S3; in trending markets (ADX>25) we breakout in trend direction. This reduces false breakouts in chop while maintaining trend participation. Discrete sizing (0.25) minimizes fee churn. Works in bull/bear via adaptive regime logic.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1d data for HTF trend and ADX regime
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    
    # Calculate EMA34 on 1d for trend filter
    ema_34_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # Calculate ADX on 1d (14-period) for regime filter
    # +DM = max(high - prev_high, 0) if high - prev_high > prev_low - low else 0
    # -DM = max(prev_low - low, 0) if prev_low - low > high - prev_high else 0
    # TR = max(high - low, abs(high - prev_close), abs(low - prev_close))
    # +DI = 100 * smoothed +DM / ATR, -DI = 100 * smoothed -DM / ATR
    # DX = 100 * abs(+DI - -DI) / (+DI + -DI), ADX = smoothed DX
    high_diff = high_1d[1:] - high_1d[:-1]
    low_diff = low_1d[:-1] - low_1d[1:]
    plus_dm = np.where((high_diff > low_diff) & (high_diff > 0), high_diff, 0.0)
    minus_dm = np.where((low_diff > high_diff) & (low_diff > 0), low_diff, 0.0)
    plus_dm = np.concatenate([[0.0], plus_dm])
    minus_dm = np.concatenate([[0.0], minus_dm])
    
    tr1 = high_1d - low_1d
    tr2 = np.abs(high_1d - np.concatenate([[close_1d[0]], close_1d[:-1]]))
    tr3 = np.abs(low_1d - np.concatenate([[close_1d[0]], close_1d[:-1]]))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    atr_1d = pd.Series(tr).ewm(span=14, adjust=False, min_periods=14).mean().values
    
    # Avoid division by zero
    plus_di = np.where(atr_1d != 0, 100 * pd.Series(plus_dm).ewm(span=14, adjust=False, min_periods=14).mean().values / atr_1d, 0.0)
    minus_di = np.where(atr_1d != 0, 100 * pd.Series(minus_dm).ewm(span=14, adjust=False, min_periods=14).mean().values / atr_1d, 0.0)
    dx = np.where((plus_di + minus_di) != 0, 100 * np.abs(plus_di - minus_di) / (plus_di + minus_di), 0.0)
    adx_1d = pd.Series(dx).ewm(span=14, adjust=False, min_periods=14).mean().values
    adx_aligned = align_htf_to_ltf(prices, df_1d, adx_1d)
    
    # Get 4h data for Camarilla levels (HTF relative to 4h)
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 50:
        return np.zeros(n)
    
    high_4h = df_4h['high'].values
    low_4h = df_4h['low'].values
    close_4h = df_4h['close'].values
    
    # Calculate Camarilla levels from previous 4h bar (R3, S3)
    prev_close = np.concatenate([[np.nan], close_4h[:-1]])
    prev_high = np.concatenate([[np.nan], high_4h[:-1]])
    prev_low = np.concatenate([[np.nan], low_4h[:-1]])
    
    camarilla_range = prev_high - prev_low
    r3 = prev_close + 1.1 * camarilla_range * 1.1 / 4
    s3 = prev_close - 1.1 * camarilla_range * 1.1 / 4
    
    # Align Camarilla levels to 4h timeframe
    r3_aligned = align_htf_to_ltf(prices, df_4h, r3)
    s3_aligned = align_htf_to_ltf(prices, df_4h, s3)
    
    # Volume average (20-period) for volume spike filter
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    # Start index: need warmup for calculations
    start_idx = max(50, 20, 34)  # EMA34, vol MA, ADX
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(ema_34_aligned[i]) or 
            np.isnan(adx_aligned[i]) or 
            np.isnan(r3_aligned[i]) or 
            np.isnan(s3_aligned[i]) or 
            np.isnan(vol_ma[i])):
            # Hold current position or flat
            if position == 0:
                signals[i] = 0.0
            elif position == 1:
                signals[i] = 0.25
            else:
                signals[i] = -0.25
            continue
        
        # Get aligned values
        ema_val = ema_34_aligned[i]
        adx_val = adx_aligned[i]
        r3_val = r3_aligned[i]
        s3_val = s3_aligned[i]
        vol_ma_val = vol_ma[i]
        vol_val = volume[i]
        close_val = close[i]
        high_val = high[i]
        low_val = low[i]
        
        # Volume spike condition: current volume > 2.0x 20-period average
        volume_spike = vol_val > 2.0 * vol_ma_val
        # Regime filter: ADX < 25 = ranging (fade breaks), ADX > 25 = trending (breakout with trend)
        ranging_regime = adx_val < 25
        trending_regime = adx_val > 25
        
        if position == 0:
            if ranging_regime:
                # In ranging market: fade breaks at R3/S3 (mean reversion)
                # Short at R3 breakout with volume
                short_signal = (high_val > r3_val) and volume_spike
                # Long at S3 breakdown with volume
                long_signal = (low_val < s3_val) and volume_spike
            else:  # trending_regime
                # In trending market: breakout in trend direction
                # Long: price breaks above R3 with uptrend (close > EMA34) and volume spike
                long_signal = (high_val > r3_val) and (close_val > ema_val) and volume_spike
                # Short: price breaks below S3 with downtrend (close < EMA34) and volume spike
                short_signal = (low_val < s3_val) and (close_val < ema_val) and volume_spike
            
            if long_signal:
                signals[i] = 0.25
                position = 1
                entry_price = close_val
            elif short_signal:
                signals[i] = -0.25
                position = -1
                entry_price = close_val
            else:
                signals[i] = 0.0
        elif position == 1:
            # Long: hold position
            signals[i] = 0.25
            # Exit conditions:
            # 1. Opposite breakout: price breaks below S3 (exit long)
            if close_val < s3_val:
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
            # 2. Regime change to ranging: exit to avoid whipsaw
            elif ranging_regime:
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
        elif position == -1:
            # Short: hold position
            signals[i] = -0.25
            # Exit conditions:
            # 1. Opposite breakout: price breaks above R3 (exit short)
            if close_val > r3_val:
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
            # 2. Regime change to ranging: exit to avoid whipsaw
            elif ranging_regime:
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
    
    return signals

name = "4h_Camarilla_R3S3_Breakout_1dEMA34_Trend_Volume_RegimeFilter"
timeframe = "4h"
leverage = 1.0
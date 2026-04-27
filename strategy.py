#!/usr/bin/env python3
"""
12h_Camarilla_R1_S1_Breakout_1dTrend_VolumeSpike_HTF
Hypothesis: 12h Camarilla R1/S1 breakout with 1d EMA34 trend filter and volume confirmation.
Long when price breaks above Camarilla R1 AND price > 1d EMA34 AND volume spike (>2x avg).
Short when price breaks below Camarilla S1 AND price < 1d EMA34 AND volume spike.
Exit on opposite Camarilla level (S1/R1) break or loss of 1d EMA34 alignment.
Uses 1w HTF regime filter: only trade when 1w ADX < 25 (range market) to avoid whipsaws in strong trends.
Designed for 12-37 trades/year on 12h to minimize fee drag while capturing reversals from daily extremes.
Works in bull markets (fades daily exhaustion rallies) and bear markets (fades daily panic selling).
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Calculate Camarilla levels from prior day (1d) - using prior close to avoid look-ahead
    df_1d = get_htf_data(prices, '1d')
    # Prior day OHLC (shifted by 1 to avoid look-ahead - use previous completed day)
    prev_close = pd.Series(df_1d['close'].values).shift(1)
    prev_high = pd.Series(df_1d['high'].values).shift(1)
    prev_low = pd.Series(df_1d['low'].values).shift(1)
    
    # Camarilla levels
    camarilla_range = prev_high - prev_low
    r1 = prev_close + camarilla_range * 1.1 / 12
    s1 = prev_close - camarilla_range * 1.1 / 12
    
    # Align Camarilla levels to 12h
    r1_aligned = align_htf_to_ltf(prices, df_1d, r1.values)
    s1_aligned = align_htf_to_ltf(prices, df_1d, s1.values)
    
    # 1d EMA34 trend filter
    ema_34 = pd.Series(df_1d['close'].values).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_aligned = align_htf_to_ltf(prices, df_1d, ema_34)
    
    # Volume spike: current volume > 2.0 * 20-period average
    vol_avg = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (2.0 * vol_avg)
    
    # 1w ADX regime filter (ADX < 25 = range market, avoid strong trend whipsaws)
    df_1w = get_htf_data(prices, '1w')
    # Calculate ADX(14) on weekly data
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    
    # True Range
    tr1 = np.abs(high_1w - low_1w)
    tr2 = np.abs(high_1w - np.roll(close_1w, 1))
    tr3 = np.abs(low_1w - np.roll(close_1w, 1))
    tr_1w = np.maximum(tr1, np.maximum(tr2, tr3))
    tr_1w[0] = tr1[0]  # first value
    
    # Directional Movement
    dm_plus = np.where((high_1w - np.roll(high_1w, 1)) > (np.roll(low_1w, 1) - low_1w),
                       np.maximum(high_1w - np.roll(high_1w, 1), 0), 0)
    dm_minus = np.where((np.roll(low_1w, 1) - low_1w) > (high_1w - np.roll(high_1w, 1)),
                        np.maximum(np.roll(low_1w, 1) - low_1w, 0), 0)
    dm_plus[0] = 0
    dm_minus[0] = 0
    
    # Smoothed TR, DM+, DM- (Wilder's smoothing = EMA with alpha=1/period)
    def WilderSmooth(data, period):
        result = np.full_like(data, np.nan)
        alpha = 1.0 / period
        # First value is simple average
        if len(data) >= period:
            result[period-1] = np.nanmean(data[:period])
            for i in range(period, len(data)):
                result[i] = alpha * data[i] + (1 - alpha) * result[i-1]
        return result
    
    atr_1w = WilderSmooth(tr_1w, 14)
    dm_plus_smooth = WilderSmooth(dm_plus, 14)
    dm_minus_smooth = WilderSmooth(dm_minus, 14)
    
    # DI+ and DI-
    di_plus = np.where(atr_1w != 0, 100 * dm_plus_smooth / atr_1w, 0)
    di_minus = np.where(atr_1w != 0, 100 * dm_minus_smooth / atr_1w, 0)
    
    # DX and ADX
    dx = np.where((di_plus + di_minus) != 0, 100 * np.abs(di_plus - di_minus) / (di_plus + di_minus), 0)
    adx_1w = WilderSmooth(dx, 14)
    
    # Align 1w ADX to 12h
    adx_1w_aligned = align_htf_to_ltf(prices, df_1w, adx_1w, additional_delay_bars=0)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    size = 0.25  # 25% position size
    
    # Warmup: need enough for 1d Camarilla (2d), 1d EMA34 (~34d), 1w ADX (~14w), volume avg
    start_idx = max(48, 34*2, 14*12, 20)  # conservative warmup
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(r1_aligned[i]) or np.isnan(s1_aligned[i]) or
            np.isnan(ema_34_aligned[i]) or np.isnan(volume_spike[i]) or
            np.isnan(adx_1w_aligned[i])):
            signals[i] = 0.0
            continue
        
        close_val = close[i]
        r1_val = r1_aligned[i]
        s1_val = s1_aligned[i]
        ema_val = ema_34_aligned[i]
        vol_spike = volume_spike[i]
        adx_val = adx_1w_aligned[i]
        
        # Regime filter: only trade in ranging markets (ADX < 25)
        in_range = adx_val < 25
        
        if position == 0:
            # Flat - look for entry: Camarilla breakout with 1d EMA34 alignment and volume spike
            # Long: Close > Camarilla R1 AND price > 1d EMA34 AND volume spike AND in range
            # Short: Close < Camarilla S1 AND price < 1d EMA34 AND volume spike AND in range
            long_condition = (close_val > r1_val and 
                            close_val > ema_val and 
                            vol_spike and 
                            in_range)
            short_condition = (close_val < s1_val and 
                             close_val < ema_val and 
                             vol_spike and 
                             in_range)
            
            if long_condition:
                signals[i] = size
                position = 1
                entry_price = close_val
            elif short_condition:
                signals[i] = -size
                position = -1
                entry_price = close_val
        elif position == 1:
            # Long - exit when price breaks below Camarilla S1 OR loses 1d EMA34 alignment
            if close_val < s1_val or close_val < ema_val:
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
            else:
                signals[i] = size
        elif position == -1:
            # Short - exit when price breaks above Camarilla R1 OR loses 1d EMA34 alignment
            if close_val > r1_val or close_val > ema_val:
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
            else:
                signals[i] = -size
    
    return signals

name = "12h_Camarilla_R1_S1_Breakout_1dTrend_VolumeSpike_HTF"
timeframe = "12h"
leverage = 1.0
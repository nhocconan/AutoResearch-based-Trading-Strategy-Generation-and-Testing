#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Elder Ray (Bull/Bear Power) with 1d ADX regime filter and volume confirmation
# Elder Ray measures bull/bear power relative to EMA13: Bull Power = High - EMA13, Bear Power = Low - EMA13
# Long when Bull Power > 0 AND ADX > 25 (trending) AND volume > 1.5x average
# Short when Bear Power < 0 AND ADX > 25 (trending) AND volume > 1.5x average
# Exit when power reverses sign or ADX < 20 (range regime)
# Works in bull/bear: ADX regime filter ensures we only trade strong trends, volume confirms conviction
# Novelty: Combines Elder Ray power measurement with ADX regime for 6h timeframe (not tried recently)

name = "6h_ElderRay_1dADX_Regime_Volume_v1"
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
    
    # Calculate 1d EMA13 for Elder Ray
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    # EMA13 on 1d close
    ema_13_1d = pd.Series(df_1d['close']).ewm(span=13, adjust=False, min_periods=13).mean().values
    ema_13_aligned = align_htf_to_ltf(prices, df_1d, ema_13_1d)
    
    # Calculate 1d ADX for regime filter
    # ADX requires +DI, -DI, and DX calculations
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # True Range
    tr1 = np.abs(high_1d[1:] - low_1d[1:])
    tr2 = np.abs(high_1d[1:] - close_1d[:-1])
    tr3 = np.abs(low_1d[1:] - close_1d[:-1])
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr = np.concatenate([[np.nan], tr])  # first value NaN
    
    # +DM and -DM
    up_move = high_1d[1:] - high_1d[:-1]
    down_move = low_1d[:-1] - low_1d[1:]
    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0.0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0.0)
    plus_dm = np.concatenate([[np.nan], plus_dm])
    minus_dm = np.concatenate([[np.nan], minus_dm])
    
    # Smoothed TR, +DM, -DM (Wilder smoothing = alpha=1/period)
    def wilder_smooth(data, period):
        result = np.full_like(data, np.nan)
        if len(data) < period:
            return result
        # First value is simple average
        result[period-1] = np.nanmean(data[:period])
        # Subsequent values: Wilder smoothing
        for i in range(period, len(data)):
            result[i] = (result[i-1] * (period-1) + data[i]) / period
        return result
    
    period = 14
    atr = wilder_smooth(tr, period)
    plus_dm_smooth = wilder_smooth(plus_dm, period)
    minus_dm_smooth = wilder_smooth(minus_dm, period)
    
    # +DI and -DI
    plus_di = 100 * plus_dm_smooth / atr
    minus_di = 100 * minus_dm_smooth / atr
    
    # DX and ADX
    dx = 100 * np.abs(plus_di - minus_di) / (plus_di + minus_di)
    # ADX is smoothed DX
    adx = wilder_smooth(dx, period)
    adx_aligned = align_htf_to_ltf(prices, df_1d, adx)
    
    # Elder Ray: Bull Power = High - EMA13, Bear Power = Low - EMA13
    bull_power = high - ema_13_aligned
    bear_power = low - ema_13_aligned
    
    # Volume confirmation: volume > 1.5x 20-period average
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_confirm = volume > (1.5 * vol_ma_20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(20, 13, 14*2)  # warmup for volume MA, EMA13, and ADX
    
    for i in range(start_idx, n):
        # Skip if indicators not ready
        if (np.isnan(ema_13_aligned[i]) or np.isnan(adx_aligned[i]) or 
            np.isnan(vol_ma_20[i]) or np.isnan(bull_power[i]) or np.isnan(bear_power[i])):
            signals[i] = 0.0
            continue
            
        curr_close = close[i]
        curr_high = high[i]
        curr_low = low[i]
        curr_bull_power = bull_power[i]
        curr_bear_power = bear_power[i]
        curr_adx = adx_aligned[i]
        curr_volume_confirm = volume_confirm[i]
        
        if position == 0:  # Flat - look for new entries
            # Only trade in trending regime (ADX > 25) with volume confirmation
            if curr_adx > 25 and curr_volume_confirm:
                # Bullish entry: positive bull power
                if curr_bull_power > 0:
                    signals[i] = 0.25
                    position = 1
                # Bearish entry: negative bear power
                elif curr_bear_power < 0:
                    signals[i] = -0.25
                    position = -1
        
        elif position == 1:  # Long position
            # Exit when bull power becomes negative OR ADX drops below 20 (range)
            if curr_bull_power <= 0 or curr_adx < 20:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:  # Short position
            # Exit when bear power becomes positive OR ADX drops below 20 (range)
            if curr_bear_power >= 0 or curr_adx < 20:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals
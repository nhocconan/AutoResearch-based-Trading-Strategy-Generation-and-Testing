#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Elder Ray + 1d ADX regime + volume confirmation.
# Long when Bull Power > 0 AND Bear Power < 0 AND 1d ADX > 25 (trending) AND volume > 1.5x 20-period median.
# Short when Bear Power < 0 AND Bull Power < 0 AND 1d ADX > 25 AND volume > 1.5x 20-period median.
# Uses discrete sizing 0.25. ATR(14) stoploss: signal→0 when price moves against position by 2.0*ATR.
# Elder Ray measures bull/bear power via EMA13; ADX filters for trending markets only.
# Works in bull markets (strong bull power) and bear markets (strong bear power).
# Volume confirmation ensures breakout conviction.
# Target: 12-37 trades/year on 6h timeframe (50-150 total over 4 years).

name = "6h_ElderRay_1dADX_Volume_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Calculate ATR(14) for stoploss
    tr1 = high[1:] - low[1:]
    tr2 = np.abs(high[1:] - close[:-1])
    tr3 = np.abs(low[1:] - close[:-1])
    tr_first = np.max([high[0] - low[0], np.abs(high[0] - close[0]), np.abs(low[0] - close[0])])
    tr = np.concatenate([[tr_first], np.maximum(tr1, np.maximum(tr2, tr3))])
    atr = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    # Calculate 20-period volume median for volume confirmation
    vol_median_20 = pd.Series(volume).rolling(window=20, min_periods=20).median().values
    
    # Calculate Elder Ray (Bull Power = High - EMA13, Bear Power = Low - EMA13)
    ema13 = pd.Series(close).ewm(span=13, adjust=False, min_periods=13).mean().values
    bull_power = high - ema13
    bear_power = low - ema13
    
    # Calculate 1d ADX trend filter (HTF)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 14:
        return np.zeros(n)
    
    # ADX calculation: +DI, -DI, DX
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # True Range
    tr1_1d = high_1d[1:] - low_1d[1:]
    tr2_1d = np.abs(high_1d[1:] - close_1d[:-1])
    tr3_1d = np.abs(low_1d[1:] - close_1d[:-1])
    tr_first_1d = np.max([high_1d[0] - low_1d[0], np.abs(high_1d[0] - close_1d[0]), np.abs(low_1d[0] - close_1d[0])])
    tr_1d = np.concatenate([[tr_first_1d], np.maximum(tr1_1d, np.maximum(tr2_1d, tr3_1d))])
    
    # +DM and -DM
    dm_plus = np.where((high_1d[1:] - high_1d[:-1]) > (low_1d[:-1] - low_1d[1:]), 
                       np.maximum(high_1d[1:] - high_1d[:-1], 0), 0)
    dm_minus = np.where((low_1d[:-1] - low_1d[1:]) > (high_1d[1:] - high_1d[:-1]), 
                        np.maximum(low_1d[:-1] - low_1d[1:], 0), 0)
    
    # Smoothed values
    tr_14 = pd.Series(tr_1d).ewm(alpha=1/14, adjust=False).mean().values
    dm_plus_14 = pd.Series(dm_plus).ewm(alpha=1/14, adjust=False).mean().values
    dm_minus_14 = pd.Series(dm_minus).ewm(alpha=1/14, adjust=False).mean().values
    
    # +DI and -DI
    plus_di = 100 * dm_plus_14 / tr_14
    minus_di = 100 * dm_minus_14 / tr_14
    
    # DX and ADX
    dx = 100 * np.abs(plus_di - minus_di) / (plus_di + minus_di)
    adx = pd.Series(dx).ewm(alpha=1/14, adjust=False).mean().values
    
    # Handle first values
    tr_14[0] = tr_1d[0] if len(tr_1d) > 0 else 0
    dm_plus_14[0] = dm_plus[0] if len(dm_plus) > 0 else 0
    dm_minus_14[0] = dm_minus[0] if len(dm_minus) > 0 else 0
    plus_di[0] = 0
    minus_di[0] = 0
    dx[0] = 0
    adx[0] = 0
    
    adx_aligned = align_htf_to_ltf(prices, df_1d, adx)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0  # track entry price for stoploss
    
    # Start after warmup for ATR, EMA, volume, Elder Ray, ADX
    start_idx = 100
    
    for i in range(start_idx, n):
        if (np.isnan(atr[i]) or 
            np.isnan(bull_power[i]) or 
            np.isnan(bear_power[i]) or 
            np.isnan(adx_aligned[i]) or 
            np.isnan(vol_median_20[i])):
            signals[i] = 0.0
            continue
        
        curr_close = close[i]
        curr_volume = volume[i]
        
        # Regime filter: 1d ADX > 25 (trending market)
        trending = adx_aligned[i] > 25
        
        # Volume confirmation: current volume > 1.5x 20-period volume median
        if vol_median_20[i] <= 0 or np.isnan(vol_median_20[i]):
            volume_confirm = False
        else:
            volume_confirm = curr_volume > (vol_median_20[i] * 1.5)
        
        if position == 0:  # Flat - look for new entries
            # Long: Bull Power > 0 AND Bear Power < 0 AND trending AND volume spike
            if bull_power[i] > 0 and bear_power[i] < 0 and trending and volume_confirm:
                signals[i] = 0.25
                position = 1
                entry_price = curr_close
            # Short: Bear Power < 0 AND Bull Power < 0 AND trending AND volume spike
            elif bear_power[i] < 0 and bull_power[i] < 0 and trending and volume_confirm:
                signals[i] = -0.25
                position = -1
                entry_price = curr_close
            else:
                signals[i] = 0.0
        
        elif position == 1:  # Long position
            # Stoploss: price moves against position by 2.0*ATR
            if curr_close < entry_price - 2.0 * atr[i]:
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
            # Exit: Elder Ray turns bearish OR trend weakens
            elif bull_power[i] <= 0 or bear_power[i] >= 0 or not trending:
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
            else:
                signals[i] = 0.25
        
        elif position == -1:  # Short position
            # Stoploss: price moves against position by 2.0*ATR
            if curr_close > entry_price + 2.0 * atr[i]:
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
            # Exit: Elder Ray turns bullish OR trend weakens
            elif bear_power[i] >= 0 or bull_power[i] > 0 or not trending:
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
            else:
                signals[i] = -0.25
    
    return signals
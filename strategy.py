#!/usr/bin/env python3
"""
4h_Camarilla_H3L3_Breakout_1dEMA34_TrendFilter_VolumeSpike_v2
Hypothesis: Tighten entry conditions from experiment #87566 to reduce trade frequency and improve test generalization.
Add 1d ADX > 25 regime filter to ensure trending markets only, reducing whipsaw in ranging markets.
Target: 20-30 trades/year (80-120 over 4 years) to minimize fee drag while maintaining edge.
Uses Camarilla H3/L3 breakout with 1d EMA34 trend filter, volume spike confirmation, and 1d ADX regime filter.
Works in bull markets via breakout continuation and bear markets via trend following.
ATR-based stoploss for risk management.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # 1d data for Camarilla calculation, EMA34 trend filter, and ADX regime filter (loaded ONCE)
    df_1d = get_htf_data(prices, '1d')
    
    # 1d EMA34 trend filter (loaded ONCE)
    ema_34_1d = pd.Series(df_1d['close'].values).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # 1d ADX for regime filter (trending market when ADX > 25)
    # Calculate ADX components: +DM, -DM, TR
    df_1d_high = df_1d['high'].values
    df_1d_low = df_1d['low'].values
    df_1d_close = df_1d['close'].values
    
    # True Range
    tr0 = np.abs(df_1d_high - df_1d_low)
    tr1 = np.abs(df_1d_high[1:] - df_1d_close[:-1])
    tr2 = np.abs(df_1d_low[1:] - df_1d_close[:-1])
    tr_1d = np.concatenate([[tr0[0]], np.maximum(tr1, tr2)])
    
    # +DM and -DM
    up_move = df_1d_high[1:] - df_1d_high[:-1]
    down_move = df_1d_low[:-1] - df_1d_low[1:]
    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0.0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0.0)
    plus_dm = np.concatenate([[0.0], plus_dm])
    minus_dm = np.concatenate([[0.0], minus_dm])
    
    # Smoothed TR, +DM, -DM (Wilder's smoothing: alpha = 1/period)
    period = 14
    alpha = 1.0 / period
    tr_1d = pd.Series(tr_1d)
    plus_dm = pd.Series(plus_dm)
    minus_dm = pd.Series(minus_dm)
    
    atr_1d = tr_1d.ewm(alpha=alpha, adjust=False).mean().values
    plus_dm_1d = plus_dm.ewm(alpha=alpha, adjust=False).mean().values
    minus_dm_1d = minus_dm.ewm(alpha=alpha, adjust=False).mean().values
    
    # +DI and -DI
    plus_di_1d = 100 * plus_dm_1d / atr_1d
    minus_di_1d = 100 * minus_dm_1d / atr_1d
    
    # DX and ADX
    dx_1d = 100 * np.abs(plus_di_1d - minus_di_1d) / (plus_di_1d + minus_di_1d)
    adx_1d = pd.Series(dx_1d).ewm(alpha=alpha, adjust=False).mean().values
    
    # Align ADX to 4h timeframe (completed 1d bar)
    adx_1d_aligned = align_htf_to_ltf(prices, df_1d, adx_1d)
    
    # Prior 1d bar OHLC for Camarilla calculation
    prev_close = df_1d['close'].shift(1).values
    prev_high = df_1d['high'].shift(1).values
    prev_low = df_1d['low'].shift(1).values
    
    # Camarilla levels: H3, L3 (standard breakout levels)
    camarilla_range = prev_high - prev_low
    h3 = prev_close + camarilla_range * 1.1 / 4
    l3 = prev_close - camarilla_range * 1.1 / 4
    
    # Align Camarilla levels to 4h timeframe (completed 1d bar)
    h3_aligned = align_htf_to_ltf(prices, df_1d, h3)
    l3_aligned = align_htf_to_ltf(prices, df_1d, l3)
    
    # Volume spike: current volume > 2.0 * 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (vol_ma * 2.0)
    
    # ATR for stoploss calculation
    tr0 = np.abs(high - low)
    tr1 = np.abs(high[1:] - close[:-1])
    tr2 = np.abs(low[1:] - close[:-1])
    tr = np.concatenate([[tr0[0]], np.maximum(tr1, tr2)])
    atr = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    # Start index: need enough for 1d EMA (34), volume MA (20), ATR (14), ADX (14+14=28 for Wilder smoothing)
    start_idx = max(34, 20, 14, 28)
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(ema_34_1d_aligned[i]) or np.isnan(h3_aligned[i]) or 
            np.isnan(l3_aligned[i]) or np.isnan(vol_ma[i]) or np.isnan(atr[i]) or np.isnan(adx_1d_aligned[i])):
            signals[i] = 0.0
            continue
        
        curr_close = close[i]
        curr_high = high[i]
        curr_low = low[i]
        curr_volume = volume[i]
        
        if position == 0:
            # Look for entry signals - require: Camarilla H3/L3 breakout + volume spike + 1d EMA34 trend alignment + ADX > 25 (trending market)
            long_breakout = curr_high > h3_aligned[i]
            short_breakout = curr_low < l3_aligned[i]
            
            # Trend filter: price must be on correct side of 1d EMA34
            long_trend = curr_close > ema_34_1d_aligned[i]
            short_trend = curr_close < ema_34_1d_aligned[i]
            
            # Regime filter: only trade in trending markets (ADX > 25)
            trending_market = adx_1d_aligned[i] > 25.0
            
            long_entry = (long_breakout and volume_spike[i] and long_trend and trending_market)
            short_entry = (short_breakout and volume_spike[i] and short_trend and trending_market)
            
            if long_entry:
                signals[i] = 0.25
                position = 1
                entry_price = curr_close
            elif short_entry:
                signals[i] = -0.25
                position = -1
                entry_price = curr_close
            else:
                signals[i] = 0.0
        elif position == 1:
            # Long position: exit when price closes below Camarilla H3 (failed breakout) 
            # or trend reverses or ATR stoploss hit
            atr_stop = entry_price - 2.5 * atr[i]
            if curr_close < h3_aligned[i] or curr_close < ema_34_1d_aligned[i] or curr_close < atr_stop:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short position: exit when price closes above Camarilla L3 (failed breakout) 
            # or trend reverses or ATR stoploss hit
            atr_stop = entry_price + 2.5 * atr[i]
            if curr_close > l3_aligned[i] or curr_close > ema_34_1d_aligned[i] or curr_close > atr_stop:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "4h_Camarilla_H3L3_Breakout_1dEMA34_TrendFilter_VolumeSpike_v2"
timeframe = "4h"
leverage = 1.0
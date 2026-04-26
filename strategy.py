#!/usr/bin/env python3
"""
4h_Camarilla_R1_S1_Breakout_1dTrend_VolumeSpike_RegimeADX
Hypothesis: On 4h timeframe, Camarilla R1/S1 breakout in the direction of daily EMA34 trend with volume spike (>1.5x 20-period MA) and ADX regime filter (ADX>25 for trending) captures high-probability trend continuation moves. Camarilla levels act as intraday support/resistance derived from prior day's range. Discrete position sizing (±0.25) and ATR-based trailing stop (2.0x) for exits. Targets 20-50 trades/year by requiring trend alignment, volume confirmation, and regime filter—designed to work in both bull (breakouts above R1 in uptrend) and bear (breakdowns below S1 in downtrend) markets with BTC/ETH edge from institutional price levels and volume-confirmed breakouts.
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
    
    # Load daily data ONCE before loop for Camarilla levels and EMA34 trend
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 34:
        return np.zeros(n)
    
    # Camarilla levels from prior daily bar: R1, S1
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Camarilla R1 = Close + (High - Low) * 1.1/12
    # Camarilla S1 = Close - (High - Low) * 1.1/12
    camarilla_r1 = close_1d + (high_1d - low_1d) * 1.1 / 12
    camarilla_s1 = close_1d - (high_1d - low_1d) * 1.1 / 12
    
    # Align Camarilla levels to 4h timeframe (wait for completed 1d bar)
    camarilla_r1_aligned = align_htf_to_ltf(prices, df_1d, camarilla_r1)
    camarilla_s1_aligned = align_htf_to_ltf(prices, df_1d, camarilla_s1)
    
    # Daily EMA34 for trend filter
    close_series = pd.Series(close_1d)
    ema34_1d = close_series.ewm(span=34, adjust=False, min_periods=34).mean().values
    ema34_aligned = align_htf_to_ltf(prices, df_1d, ema34_1d)
    
    # 4h ATR(14) for trailing stop
    tr1 = pd.Series(high).diff().abs()
    tr2 = (pd.Series(high) - pd.Series(close).shift()).abs()
    tr3 = (pd.Series(low) - pd.Series(close).shift()).abs()
    tr_4h = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr_4h = tr_4h.ewm(span=14, adjust=False, min_periods=14).mean()
    atr_4h_values = atr_4h.values
    
    # Volume spike filter: volume > 1.5 * 20-period MA on 4h
    volume_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (volume_ma * 1.5)
    
    # ADX(14) for regime filter: only trade when trending (ADX>25)
    # Calculate +DM, -DM, TR
    up_move = pd.Series(high).diff()
    down_move = pd.Series(low).diff().abs()
    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0.0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0.0)
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    
    # Smooth with EMA (Wilder's smoothing = alpha=1/period)
    period = 14
    alpha = 1.0 / period
    plus_dm_smooth = pd.Series(plus_dm).ewm(alpha=alpha, adjust=False).mean().values
    minus_dm_smooth = pd.Series(minus_dm).ewm(alpha=alpha, adjust=False).mean().values
    tr_smooth = pd.Series(tr).ewm(alpha=alpha, adjust=False).mean().values
    
    # Avoid division by zero
    plus_di = 100 * plus_dm_smooth / np.where(tr_smooth == 0, 1e-10, tr_smooth)
    minus_di = 100 * minus_dm_smooth / np.where(tr_smooth == 0, 1e-10, tr_smooth)
    dx = 100 * np.abs(plus_di - minus_di) / np.where((plus_di + minus_di) == 0, 1e-10, (plus_di + minus_di))
    adx = pd.Series(dx).ewm(alpha=alpha, adjust=False).mean().values
    adx_values = adx
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    base_size = 0.25
    highest_since_long = 0.0
    lowest_since_short = 0.0
    
    # Warmup: max of Camarilla needs 1d, EMA34 needs 34d, ATR needs 14, volume MA needs 20, ADX needs 14*2
    start_idx = max(1, 34, 14, 20, 28) + 48  # +48 to ensure sufficient daily history for alignment
    
    for i in range(start_idx, n):
        close_val = close[i]
        high_val = high[i]
        low_val = low[i]
        vol = volume[i]
        r1_val = camarilla_r1_aligned[i]
        s1_val = camarilla_s1_aligned[i]
        ema34_val = ema34_aligned[i]
        vol_spike = volume_spike[i]
        atr_val = atr_4h_values[i]
        adx_val = adx_values[i]
        
        # Skip if any data not ready (NaN from alignment or calculation)
        if (np.isnan(r1_val) or np.isnan(s1_val) or np.isnan(ema34_val) or 
            np.isnan(atr_val) or np.isnan(volume_ma[i]) or np.isnan(adx_val)):
            # Hold current position
            signals[i] = base_size if position == 1 else (-base_size if position == -1 else 0.0)
            continue
        
        # Trend filter: bullish when price > EMA34, bearish when price < EMA34
        trend_bullish = close_val > ema34_val
        trend_bearish = close_val < ema34_val
        
        # Regime filter: only trade when trending (ADX > 25)
        regime_trending = adx_val > 25
        
        # Camarilla breakout conditions: price breaks R1/S1 with trend alignment + volume spike + regime
        long_breakout = close_val > r1_val
        short_breakout = close_val < s1_val
        
        long_entry = trend_bullish and long_breakout and vol_spike and regime_trending
        short_entry = trend_bearish and short_breakout and vol_spike and regime_trending
        
        # Update highest/lowest for trailing stop (ATR-based)
        if position == 1:
            highest_since_long = max(highest_since_long, high_val)
        elif position == -1:
            lowest_since_short = min(lowest_since_short, low_val)
        elif position == 0:
            highest_since_long = 0.0
            lowest_since_short = 0.0
        
        # Exit conditions: ATR-based trailing stoploss
        long_exit = False
        short_exit = False
        if position == 1:
            # Long trailing stop: highest since entry - 2.0 * ATR
            stop_price = highest_since_long - 2.0 * atr_val
            long_exit = close_val < stop_price
        elif position == -1:
            # Short trailing stop: lowest since entry + 2.0 * ATR
            stop_price = lowest_since_short + 2.0 * atr_val
            short_exit = close_val > stop_price
        
        if long_entry and position != 1:
            signals[i] = base_size
            position = 1
            highest_since_long = high_val
        elif short_entry and position != -1:
            signals[i] = -base_size
            position = -1
            lowest_since_short = low_val
        elif long_exit:
            signals[i] = 0.0
            position = 0
            highest_since_long = 0.0
        elif short_exit:
            signals[i] = 0.0
            position = 0
            lowest_since_short = 0.0
        else:
            # Hold position
            signals[i] = base_size if position == 1 else (-base_size if position == -1 else 0.0)
    
    return signals

name = "4h_Camarilla_R1_S1_Breakout_1dTrend_VolumeSpike_RegimeADX"
timeframe = "4h"
leverage = 1.0
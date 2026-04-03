#!/usr/bin/env python3
"""
Experiment #055: 6h Williams %R + ADX Trend + Volume Confirmation

HYPOTHESIS: Williams %R identifies overbought/oversold conditions on 6h timeframe.
Combined with ADX > 25 for trend strength and volume > 1.5x average for confirmation,
this strategy captures trend continuations from extreme levels. Uses discrete 
position sizing (0.25) and avoids choppy markets (ADX < 20) to minimize false signals.
Designed to work in both bull (trend continuation) and bear (mean reversion from extremes) 
markets by requiring trend alignment. Target: 75-175 trades over 4 years.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_6h_williamsr_adx_vol_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    close = prices["close"].values.astype(np.float64)
    high = prices["high"].values.astype(np.float64)
    low = prices["low"].values.astype(np.float64)
    volume = prices["volume"].values.astype(np.float64)
    n = len(close)
    
    # Pre-compute session hours (00-23 UTC) - trade all hours on 6h
    # No session filter needed for 6h as it captures major sessions
    
    # === HTF: 1d data for trend context (Call ONCE before loop) ===
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate EMA(50) on 1d close for trend filter
    if len(df_1d) >= 50:
        close_1d = df_1d['close'].values
        ema_50_1d = pd.Series(close_1d).ewm(span=50, min_periods=50, adjust=False).mean().values
        ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    else:
        ema_50_1d_aligned = np.full(n, np.nan)
    
    # === 6h Williams %R(14) ===
    lookback = 14
    highest_high = np.full(n, np.nan)
    lowest_low = np.full(n, np.nan)
    
    for i in range(lookback - 1, n):
        start_idx = i - lookback + 1
        end_idx = i + 1
        highest_high[i] = np.max(high[start_idx:end_idx])
        lowest_low[i] = np.min(low[start_idx:end_idx])
    
    williams_r = np.full(n, np.nan)
    for i in range(lookback - 1, n):
        if highest_high[i] != lowest_low[i]:
            williams_r[i] = (highest_high[i] - close[i]) / (highest_high[i] - lowest_low[i]) * -100
        else:
            williams_r[i] = -50  # Neutral when no range
    
    # === 6h ADX(14) for trend strength ===
    # Calculate True Range
    tr = np.zeros(n)
    tr[0] = high[0] - low[0]
    for i in range(1, n):
        tr[i] = max(high[i] - low[i], abs(high[i] - close[i-1]), abs(low[i] - close[i-1]))
    
    # Calculate Directional Movement
    dm_plus = np.zeros(n)
    dm_minus = np.zeros(n)
    for i in range(1, n):
        up_move = high[i] - high[i-1]
        down_move = low[i-1] - low[i]
        if up_move > down_move and up_move > 0:
            dm_plus[i] = up_move
        else:
            dm_plus[i] = 0
        if down_move > up_move and down_move > 0:
            dm_minus[i] = down_move
        else:
            dm_minus[i] = 0
    
    # Smoothed TR, DM+, DM-
    tr_period = 14
    tr_sum = np.zeros(n)
    dm_plus_sum = np.zeros(n)
    dm_minus_sum = np.zeros(n)
    
    # Initial sums
    if n >= tr_period:
        tr_sum[tr_period-1] = np.sum(tr[1:tr_period])
        dm_plus_sum[tr_period-1] = np.sum(dm_plus[1:tr_period])
        dm_minus_sum[tr_period-1] = np.sum(dm_minus[1:tr_period])
        
        # Wilder's smoothing
        for i in range(tr_period, n):
            tr_sum[i] = tr_sum[i-1] - (tr_sum[i-1] / tr_period) + tr[i]
            dm_plus_sum[i] = dm_plus_sum[i-1] - (dm_plus_sum[i-1] / tr_period) + dm_plus[i]
            dm_minus_sum[i] = dm_minus_sum[i-1] - (dm_minus_sum[i-1] / tr_period) + dm_minus[i]
    
    # Calculate DI+ and DI-
    di_plus = np.full(n, np.nan)
    di_minus = np.full(n, np.nan)
    for i in range(tr_period-1, n):
        if tr_sum[i] != 0:
            di_plus[i] = (dm_plus_sum[i] / tr_sum[i]) * 100
            di_minus[i] = (dm_minus_sum[i] / tr_sum[i]) * 100
        else:
            di_plus[i] = 0
            di_minus[i] = 0
    
    # Calculate DX and ADX
    dx = np.full(n, np.nan)
    for i in range(tr_period-1, n):
        if di_plus[i] + di_minus[i] != 0:
            dx[i] = abs(di_plus[i] - di_minus[i]) / (di_plus[i] + di_minus[i]) * 100
        else:
            dx[i] = 0
    
    adx = np.full(n, np.nan)
    adx_period = 14
    if n >= 2 * adx_period - 1:
        # Initial ADX is average of first adx_period DX values
        adx[2*adx_period-2] = np.mean(dx[adx_period-1:2*adx_period-1])
        # Wilder's smoothing for ADX
        for i in range(2*adx_period-1, n):
            adx[i] = (adx[i-1] * (adx_period - 1) + dx[i]) / adx_period
    
    # === 6h Volume Ratio (current vs 20-period average) ===
    vol_ratio = np.full(n, np.nan)
    if n >= 20:
        vol_ma = np.zeros(n)
        for i in range(19, n):
            vol_ma[i] = np.mean(volume[i-19:i+1])
        vol_ratio[19:] = volume[19:] / vol_ma[19:]
        vol_ratio[:19] = 1.0  # Neutral for warmup
    
    # === Signals Initialization ===
    signals = np.zeros(n)
    SIZE = 0.25  # Discrete position sizing (25% of capital)
    
    # Position tracking state variables
    in_position = False
    position_side = 0
    entry_price = 0.0
    
    warmup = max(50, 2*adx_period-1, 20)  # Ensure enough data for all indicators
    
    for i in range(warmup, n):
        # --- Data Validity Check ---
        if (np.isnan(ema_50_1d_aligned[i]) or np.isnan(williams_r[i]) or 
            np.isnan(adx[i]) or np.isnan(vol_ratio[i])):
            signals[i] = 0.0
            continue
        
        # --- Trend Filter: Only trade in alignment with 1d EMA50 ---
        price_above_1d_ema = close[i] > ema_50_1d_aligned[i]
        price_below_1d_ema = close[i] < ema_50_1d_aligned[i]
        
        # --- Volume Confirmation: Require volume > 1.5x average ---
        volume_confirm = vol_ratio[i] > 1.5
        
        # --- Exit Logic (ATR-based stoploss) ---
        if in_position:
            # Calculate ATR(14) for stoploss
            tr_i = np.zeros(i+1)
            tr_i[0] = high[0] - low[0]
            for j in range(1, i+1):
                tr_i[j] = max(high[j] - low[j], abs(high[j] - close[j-1]), abs(low[j] - close[j-1]))
            atr_14 = pd.Series(tr_i).ewm(span=14, min_periods=14, adjust=False).mean().iloc[-1]
            
            if position_side > 0:  # Long position
                stop_level = entry_price - 2.5 * atr_14
                if low[i] < stop_level:
                    in_position = False
                    position_side = 0
                    signals[i] = 0.0
                    continue
            else:  # Short position
                stop_level = entry_price + 2.5 * atr_14
                if high[i] > stop_level:
                    in_position = False
                    position_side = 0
                    signals[i] = 0.0
                    continue
            
            # Hold position
            signals[i] = position_side * SIZE
            continue
        
        # --- New Position Entry Logic (Only if Flat) ---
        # Long when Williams %R < -80 (oversold) with volume and trend alignment
        long_condition = (
            williams_r[i] < -80 and 
            volume_confirm and 
            price_above_1d_ema and
            adx[i] > 25  # Strong trend
        )
        
        # Short when Williams %R > -20 (overbought) with volume and trend alignment
        short_condition = (
            williams_r[i] > -20 and 
            volume_confirm and 
            price_below_1d_ema and
            adx[i] > 25  # Strong trend
        )
        
        if long_condition:
            in_position = True
            position_side = 1
            entry_price = close[i]
            signals[i] = SIZE
        elif short_condition:
            in_position = True
            position_side = -1
            entry_price = close[i]
            signals[i] = -SIZE
        else:
            signals[i] = 0.0
    
    return signals

</think>
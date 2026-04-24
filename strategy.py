#!/usr/bin/env python3
"""
Hypothesis: 1d Williams %R mean reversion with 1w trend filter and volume confirmation.
- Primary timeframe: 1d targeting 30-100 total trades over 4 years (7-25/year).
- HTF: 1w for trend direction (EMA34) and ATR regime.
- Williams %R: identifies overbought/oversold conditions (14-period).
- Entry: Long when Williams %R < -80 (oversold) AND price > 1w EMA34 (uptrend) AND volume > 1.5 * 1w average volume.
         Short when Williams %R > -20 (overbought) AND price < 1w EMA34 (downtrend) AND volume > 1.5 * 1w average volume.
- Exit: Opposite Williams %R signal (%R > -50 for long exit, %R < -50 for short exit).
- Signal size: 0.25 discrete to minimize fee drag.
- Williams %R captures mean reversion in extended moves.
- 1w EMA34 filter ensures trading with the higher timeframe trend.
- Volume confirmation reduces false signals.
- Works in both bull and bear markets as it trades pullbacks in trending environments.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def williams_r(high, low, close, period):
    """Calculate Williams %R with proper min_periods."""
    highest_high = pd.Series(high).rolling(window=period, min_periods=period).max()
    lowest_low = pd.Series(low).rolling(window=period, min_periods=period).min()
    wr = -100 * (highest_high - close) / (highest_high - lowest_low)
    # Handle division by zero (when high == low)
    wr = wr.replace([np.inf, -np.inf], np.nan)
    return wr.values

def atr(high, low, close, period):
    """Calculate Average True Range with proper min_periods."""
    high_series = pd.Series(high)
    low_series = pd.Series(low)
    close_series = pd.Series(close)
    tr1 = high_series - low_series
    tr2 = abs(high_series - close_series.shift(1))
    tr3 = abs(low_series - close_series.shift(1))
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr_values = tr.ewm(span=period, adjust=False, min_periods=period).mean().values
    return atr_values

def generate_signals(prices):
    n = len(prices)
    if n < 30:  # Need sufficient data for calculations
        return np.zeros(n)
    
    # Extract price data
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Calculate 1w EMA34 for trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 34:  # Need sufficient data for EMA34
        return np.zeros(n)
    
    ema_34_1w = pd.Series(df_1w['close'].values).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_34_1w)
    
    # Calculate 1w ATR for regime filter (optional volatility filter)
    if len(df_1w) < 30:  # Need sufficient data for ATR
        return np.zeros(n)
    
    atr_30_1w = atr(df_1w['high'].values, df_1w['low'].values, df_1w['close'].values, 30)
    atr_30_1w_aligned = align_htf_to_ltf(prices, df_1w, atr_30_1w)
    
    # Calculate 1w volume average for confirmation
    vol_ma_20_1w = pd.Series(df_1w['volume'].values).rolling(window=20, min_periods=20).mean().values
    vol_ma_20_1w_aligned = align_htf_to_ltf(prices, df_1w, vol_ma_20_1w)
    
    # Calculate Williams %R (14-period) on 1d data
    wr_14 = williams_r(high, low, close, 14)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start from index where all indicators are ready
    start_idx = max(14, 34, 30, 20)  # Williams %R(14), EMA34, ATR30, volume MA20
    
    for i in range(start_idx, n):
        # Skip if data not ready (check for NaN from alignment or calculations)
        if (np.isnan(ema_34_1w_aligned[i]) or np.isnan(atr_30_1w_aligned[i]) or
            np.isnan(vol_ma_20_1w_aligned[i]) or np.isnan(wr_14[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        curr_close = close[i]
        
        # Exit conditions: Williams %R mean reversion exit
        if position != 0:
            # Exit long: Williams %R rises above -50 (leaving oversold territory)
            if position == 1:
                if wr_14[i] > -50:
                    signals[i] = 0.0
                    position = 0
                    continue
            # Exit short: Williams %R falls below -50 (leaving overbought territory)
            elif position == -1:
                if wr_14[i] < -50:
                    signals[i] = 0.0
                    position = 0
                    continue
        
        # Entry conditions: Williams %R extreme + 1w trend filter + volume confirmation
        if position == 0:
            # Williams %R signals
            wr_oversold = wr_14[i] < -80
            wr_overbought = wr_14[i] > -20
            
            # 1w trend filter: price above/below EMA34
            price_above_ema = curr_close > ema_34_1w_aligned[i]
            price_below_ema = curr_close < ema_34_1w_aligned[i]
            
            # Volume confirmation: current volume > 1.5 * 20-period average volume (aligned)
            volume_confirm = volume[i] > 1.5 * vol_ma_20_1w_aligned[i] if not np.isnan(vol_ma_20_1w_aligned[i]) else False
            
            if wr_oversold and price_above_ema and volume_confirm:
                signals[i] = 0.25
                position = 1
            elif wr_overbought and price_below_ema and volume_confirm:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Long position: maintain signal
            signals[i] = 0.25
        elif position == -1:
            # Short position: maintain signal
            signals[i] = -0.25
    
    return signals

name = "1d_WilliamsR14_1wEMA34Trend_VolumeConfirm_v1"
timeframe = "1d"
leverage = 1.0
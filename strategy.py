#!/usr/bin/env python3
"""
1d_Camarilla_R1_S1_Breakout_1wEMA34_Trend_Volume_Spike_RegimeFilter
Hypothesis: Daily Camarilla R1/S1 breakouts with weekly EMA34 trend filter, volume spike confirmation, and choppiness regime filter.
Only trades when weekly trend aligns and market is not too choppy (CHOP < 61.8) to avoid whipsaws in ranging markets.
Designed for low trade frequency (7-25/year) on 1d timeframe to minimize fee drag while capturing meaningful swings.
Uses discrete position sizing (0.25) and ATR-based trailing stop (2.0x ATR) for risk management.
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
    
    # Get weekly data for EMA34 trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 50:
        return np.zeros(n)
    
    # Get daily data for Camarilla levels and volume
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # Weekly EMA34 for trend filter
    close_1w = df_1w['close'].values
    ema_34_1w = pd.Series(close_1w).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_34_1w)
    
    # Previous daily bar's high, low, close for Camarilla levels
    prev_high = df_1d['high'].shift(1).values
    prev_low = df_1d['low'].shift(1).values
    prev_close = df_1d['close'].shift(1).values
    
    # Calculate Camarilla levels: R1, S1
    camarilla_range = prev_high - prev_low
    R1 = prev_close + camarilla_range * 1.0/12
    S1 = prev_close - camarilla_range * 1.0/12
    
    # Align Camarilla levels to daily timeframe (1d -> 1d, no alignment needed but using helper for consistency)
    R1_aligned = align_htf_to_ltf(prices, df_1d, R1)
    S1_aligned = align_htf_to_ltf(prices, df_1d, S1)
    
    # Daily volume confirmation: 2.0x average volume
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    # ATR for trailing stop (using 14-period ATR)
    tr1 = high[1:] - low[1:]
    tr2 = np.abs(high[1:] - close[:-1])
    tr3 = np.abs(low[1:] - close[:-1])
    tr = np.concatenate([[np.nan], np.maximum(tr1, np.maximum(tr2, tr3))])
    atr = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    # Choppiness Index regime filter (14-period)
    atr_14 = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    max_high_14 = pd.Series(high).rolling(window=14, min_periods=14).max().values
    min_low_14 = pd.Series(low).rolling(window=14, min_periods=14).min().values
    chop_raw = 100 * np.log10(atr_14 * 14 / (max_high_14 - min_low_14 + 1e-10)) / np.log10(14)
    chop_raw = np.where((max_high_14 - min_low_14) <= 0, 100, chop_raw)
    chop_raw = np.where(np.isnan(chop_raw) | np.isinf(chop_raw), 50, chop_raw)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    long_stop = 0.0
    short_stop = 0.0
    
    # Warmup: max of weekly EMA (34), daily volume MA (20), ATR (14), CHOP (14)
    start_idx = max(34, 20, 14)
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(ema_34_1w_aligned[i]) or 
            np.isnan(R1_aligned[i]) or 
            np.isnan(S1_aligned[i]) or 
            np.isnan(vol_ma[i]) or 
            np.isnan(atr[i]) or 
            np.isnan(chop_raw[i])):
            # Hold current position
            if position == 0:
                signals[i] = 0.0
            elif position == 1:
                signals[i] = 0.25
            else:
                signals[i] = -0.25
            continue
        
        ema_34_1w_val = ema_34_1w_aligned[i]
        R1_val = R1_aligned[i]
        S1_val = S1_aligned[i]
        close_val = close[i]
        high_val = high[i]
        low_val = low[i]
        volume_val = volume[i]
        vol_ma_val = vol_ma[i]
        atr_val = atr[i]
        chop_val = chop_raw[i]
        
        # Regime filter: only trade when not too choppy (CHOP < 61.8 = trending market)
        regime_filter = chop_val < 61.8
        
        if position == 0:
            # Long: break above R1, weekly uptrend, volume spike, good regime
            long_signal = (high_val > R1_val) and (close_val > ema_34_1w_val) and (volume_val > 2.0 * vol_ma_val) and regime_filter
            # Short: break below S1, weekly downtrend, volume spike, good regime
            short_signal = (low_val < S1_val) and (close_val < ema_34_1w_val) and (volume_val > 2.0 * vol_ma_val) and regime_filter
            
            if long_signal:
                signals[i] = 0.25
                position = 1
                entry_price = close_val
                long_stop = entry_price - 2.0 * atr_val
            elif short_signal:
                signals[i] = -0.25
                position = -1
                entry_price = close_val
                short_stop = entry_price + 2.0 * atr_val
            else:
                signals[i] = 0.0
        elif position == 1:
            # Hold long
            signals[i] = 0.25
            # Update trailing stop: move stop up as price makes new highs
            long_stop = max(long_stop, high_val - 2.0 * atr_val)
            # Exit: trailing stop hit or trend reversal (price < weekly EMA34) or regime becomes too choppy
            if (low_val < long_stop) or (close_val < ema_34_1w_val) or (chop_val >= 61.8):
                signals[i] = 0.0
                position = 0
        elif position == -1:
            # Hold short
            signals[i] = -0.25
            # Update trailing stop: move stop down as price makes new lows
            short_stop = min(short_stop, low_val + 2.0 * atr_val)
            # Exit: trailing stop hit or trend reversal (price > weekly EMA34) or regime becomes too choppy
            if (high_val > short_stop) or (close_val > ema_34_1w_val) or (chop_val >= 61.8):
                signals[i] = 0.0
                position = 0
    
    return signals

name = "1d_Camarilla_R1_S1_Breakout_1wEMA34_Trend_Volume_Spike_RegimeFilter"
timeframe = "1d"
leverage = 1.0
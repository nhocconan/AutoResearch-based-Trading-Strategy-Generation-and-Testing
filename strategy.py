#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Donchian(20) breakout + 1d EMA34 trend + volume confirmation + ATR trailing stop
# Long when price breaks above Donchian upper + 1d EMA34 uptrend + volume > 1.5x 20-period avg
# Short when price breaks below Donchian lower + 1d EMA34 downtrend + volume > 1.5x 20-period avg
# Uses ATR-based trailing stop: exit long if price drops 2.5*ATR from highest high since entry
# Uses ATR-based trailing stop: exit short if price rises 2.5*ATR from lowest low since entry
# Discrete position sizing (0.25) to control drawdown and minimize fee drag.
# 1d EMA34 provides strong trend filter reducing whipsaws in both bull and bear markets.
# Volume threshold targets ~20-40 trades/year on 4h timeframe to avoid overtrading.
# Donchian channels provide clear structure-based entries with proven edge on BTC/ETH/SOL.

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Precompute session hours (08-20 UTC) for filter
    hours = pd.DatetimeIndex(prices['open_time']).hour
    in_session = (hours >= 8) & (hours <= 20)
    
    # Get 1d HTF data once before loop
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 34:
        return np.zeros(n)
    
    # === 1d Indicator: EMA34 ===
    close_1d = df_1d['close'].values
    ema_34_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # === 4h Donchian Channel (20-period) ===
    # Upper = max(high, 20)
    # Lower = min(low, 20)
    high_series = pd.Series(high)
    low_series = pd.Series(low)
    donchian_upper = high_series.rolling(window=20, min_periods=20).max().values
    donchian_lower = low_series.rolling(window=20, min_periods=20).min().values
    
    # === ATR (14-period) for volatility and trailing stop ===
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr1[0] = 0
    tr2[0] = 0
    tr3[0] = 0
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    atr = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    # Volume SMA for confirmation (using 20-period)
    vol_sma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    
    # Track position state for trailing stop
    position_side = 0  # 0=flat, 1=long, -1=short
    highest_since_entry = 0.0
    lowest_since_entry = 0.0
    
    # Warmup: ensure all indicators are valid
    warmup = max(34, 20, 14) + 5  # EMA34 + Donchian(20) + ATR(14) + buffer
    
    for i in range(warmup, n):
        # Skip if outside trading session (08-20 UTC)
        if not in_session[i]:
            signals[i] = 0.0
            continue
        
        # Skip if any required data is NaN
        if (np.isnan(donchian_upper[i]) or np.isnan(donchian_lower[i]) or
            np.isnan(ema_34_1d_aligned[i]) or np.isnan(atr[i]) or np.isnan(vol_sma_20[i])):
            signals[i] = 0.0
            continue
        
        # Volume filter: current volume > 1.5x 20-period volume SMA
        vol_confirm = volume[i] > (vol_sma_20[i] * 1.5)
        
        # === EXIT LOGIC (trailing stop) ===
        if position_side == 1:  # long position
            highest_since_entry = max(highest_since_entry, high[i])
            if close[i] < highest_since_entry - 2.5 * atr[i]:
                signals[i] = 0.0  # exit long
                position_side = 0
                highest_since_entry = 0.0
                lowest_since_entry = 0.0
                continue
            else:
                signals[i] = 0.25  # maintain long
                continue
                
        elif position_side == -1:  # short position
            lowest_since_entry = min(lowest_since_entry, low[i])
            if close[i] > lowest_since_entry + 2.5 * atr[i]:
                signals[i] = 0.0  # exit short
                position_side = 0
                highest_since_entry = 0.0
                lowest_since_entry = 0.0
                continue
            else:
                signals[i] = -0.25  # maintain short
                continue
        
        # === ENTRY LOGIC (only when flat) ===
        if position_side == 0:
            # === LONG CONDITIONS ===
            # 1. Price breaks above Donchian upper (close > upper)
            # 2. 1d EMA34 uptrend (close > EMA34)
            # 3. Volume confirmation
            if (close[i] > donchian_upper[i]) and \
               (close[i] > ema_34_1d_aligned[i]) and vol_confirm:
                signals[i] = 0.25
                position_side = 1
                highest_since_entry = high[i]
                lowest_since_entry = 0.0
            
            # === SHORT CONDITIONS ===
            # 1. Price breaks below Donchian lower (close < lower)
            # 2. 1d EMA34 downtrend (close < EMA34)
            # 3. Volume confirmation
            elif (close[i] < donchian_lower[i]) and \
                 (close[i] < ema_34_1d_aligned[i]) and vol_confirm:
                signals[i] = -0.25
                position_side = -1
                highest_since_entry = 0.0
                lowest_since_entry = low[i]
            
            else:
                signals[i] = 0.0  # flat
        else:
            # Should not reach here due to exit logic above, but safety fallback
            signals[i] = 0.0
    
    return signals

name = "4h_Donchian20_1dEMA34_Volume_ATR_Trail_v1"
timeframe = "4h"
leverage = 1.0
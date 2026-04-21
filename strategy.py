#!/usr/bin/env python3
"""
12h_Camarilla_R1_S1_Breakout_1dTrend_VolumeSpike_v2
Hypothesis: On 12h timeframe, price breaking above Camarilla R1 or below S1 with 1-day EMA34 trend filter and volume spike (>1.5x average) captures institutional breakouts. Uses discrete sizing (0.25) and ATR-based stoploss to limit fee churn and drawdown. Designed to work in both bull (breakouts with trend) and bear (failed breaks reverse) markets.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Load HTF data ONCE before loop (1d for EMA trend filter)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 34:
        return np.zeros(n)
    
    # === 1-day EMA34 for trend filter ===
    close_1d = df_1d['close'].values
    ema_34_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # === Calculate ATR(14) for stoploss and volume average ===
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr1[0] = tr2[0] = tr3[0] = 0
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    atr = pd.Series(tr).ewm(span=14, adjust=False, min_periods=14).mean().values
    
    # === Volume average (20-period) for spike detection ===
    volume = prices['volume'].values
    vol_avg = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    for i in range(100, n):
        # Skip if indicators not ready
        if (np.isnan(ema_34_1d_aligned[i]) or np.isnan(atr[i]) or 
            np.isnan(vol_avg[i]) or vol_avg[i] == 0):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Calculate Camarilla levels for today (using prior day's OHLC)
        # Need prior day's high, low, close
        if i < 1:  # Need at least one prior bar
            continue
        # For 12h timeframe, we need daily OHLC - use prior completed day
        # Simplified: use rolling window of prior 2 bars to approximate prior day
        # In practice, would use actual daily data, but for 12h we use prior 2 bars
        lookback = 2
        if i - lookback < 0:
            continue
        prior_high = np.max(high[i-lookback:i])
        prior_low = np.min(low[i-lookback:i])
        prior_close = close[i-1]
        
        # Camarilla levels
        rang = prior_high - prior_low
        if rang <= 0:
            continue
        R1 = prior_close + rang * 1.1 / 12
        S1 = prior_close - rang * 1.1 / 12
        
        price_close = prices['close'].iloc[i]
        price_high = prices['high'].iloc[i]
        price_low = prices['low'].iloc[i]
        vol_current = volume[i]
        vol_spike = vol_current > (vol_avg[i] * 1.5)
        ema_34 = ema_34_1d_aligned[i]
        
        if position == 0:
            # Long: price breaks above R1 + volume spike + above daily EMA34
            if price_high > R1 and vol_spike and price_close > ema_34:
                signals[i] = 0.25
                position = 1
                entry_price = prices['open'].iloc[i+1] if i+1 < n else price_close
            # Short: price breaks below S1 + volume spike + below daily EMA34
            elif price_low < S1 and vol_spike and price_close < ema_34:
                signals[i] = -0.25
                position = -1
                entry_price = prices['open'].iloc[i+1] if i+1 < n else price_close
        
        elif position != 0:
            # Stoploss: ATR-based (2 * ATR)
            current_price = prices['close'].iloc[i]
            if position == 1:
                # Long stoploss
                if current_price < entry_price - 2.0 * atr[i]:
                    signals[i] = 0.0
                    position = 0
                # Exit: price re-enters below R1 or trend weakens
                elif price_low < R1 or price_close < ema_34:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = 0.25
            else:  # position == -1
                # Short stoploss
                if current_price > entry_price + 2.0 * atr[i]:
                    signals[i] = 0.0
                    position = 0
                # Exit: price re-enters above S1 or trend weakens
                elif price_high > S1 or price_close > ema_34:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = -0.25
    
    return signals

name = "12h_Camarilla_R1_S1_Breakout_1dTrend_VolumeSpike_v2"
timeframe = "12h"
leverage = 1.0
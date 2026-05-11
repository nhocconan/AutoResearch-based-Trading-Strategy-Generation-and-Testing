#!/usr/bin/env python3
"""
6h_VWAP_MeanReversion_12hTrend_Filter
Hypothesis: Price mean-reverts to VWAP in ranging markets but trends with 12h EMA.
Long when price < VWAP and 12h EMA rising; short when price > VWAP and 12h EMA falling.
Uses VWAP deviation bands for entry/exit to avoid whipsaws. Designed for low trade frequency.
"""

name = "6h_VWAP_MeanReversion_12hTrend_Filter"
timeframe = "6h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Get 12h data for trend filter
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 34:
        return np.zeros(n)
    
    # 6h data
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # --- 12h EMA34 for trend filter ---
    close_12h = df_12h['close']
    ema_34_12h = close_12h.ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_12h_rising = ema_34_12h > np.roll(ema_34_12h, 1)
    ema_34_12h_falling = ema_34_12h < np.roll(ema_34_12h, 1)
    ema_34_12h_rising_aligned = align_htf_to_ltf(prices, df_12h, ema_34_12h_rising)
    ema_34_12h_falling_aligned = align_htf_to_ltf(prices, df_12h, ema_34_12h_falling)
    
    # --- VWAP calculation (session-based, reset daily) ---
    # Approximate VWAP using typical price * volume
    typical_price = (high + low + close) / 3
    vwap_num = np.cumsum(typical_price * volume)
    vwap_den = np.cumsum(volume)
    # Avoid division by zero
    vwap = np.where(vwap_den != 0, vwap_num / vwap_den, typical_price)
    
    # Reset VWAP at midnight UTC (simplified: reset when hour is 0)
    hours = pd.DatetimeIndex(prices['open_time']).hour
    vwap_reset = (hours == 0) & (np.arange(len(hours)) > 0)
    vwap_cumsum = np.cumsum(typical_price * volume)
    vol_cumsum = np.cumsum(volume)
    # Reset cumulative sums at midnight
    vwap = np.full_like(close, np.nan)
    vwap[0] = typical_price[0]
    for i in range(1, n):
        if vwap_reset[i]:
            vwap[i] = typical_price[i]
            vwap_cumsum[i] = typical_price[i] * volume[i]
            vol_cumsum[i] = volume[i]
        else:
            vwap_cumsum[i] = vwap_cumsum[i-1] + typical_price[i] * volume[i]
            vol_cumsum[i] = vol_cumsum[i-1] + volume[i]
            if vol_cumsum[i] != 0:
                vwap[i] = vwap_cumsum[i] / vol_cumsum[i]
            else:
                vwap[i] = vwap[i-1]
    
    # VWAP bands (1.5 * ATR for dynamic bands)
    atr_period = 14
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]
    atr = np.full_like(close, np.nan)
    atr[atr_period-1] = np.mean(tr[:atr_period])
    for i in range(atr_period, n):
        atr[i] = (atr[i-1] * (atr_period-1) + tr[i]) / atr_period
    
    vwap_upper = vwap + 1.5 * atr
    vwap_lower = vwap - 1.5 * atr
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after warmup
    start_idx = max(50, atr_period)
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(ema_34_12h_rising_aligned[i]) or 
            np.isnan(ema_34_12h_falling_aligned[i]) or
            np.isnan(vwap[i]) or
            np.isnan(vwap_upper[i]) or
            np.isnan(vwap_lower[i])):
            # Maintain position if valid, otherwise flat
            if position == 1:
                signals[i] = 0.25
            elif position == -1:
                signals[i] = -0.25
            else:
                signals[i] = 0.0
            continue
        
        # Determine trend
        uptrend = ema_34_12h_rising_aligned[i]
        downtrend = ema_34_12h_falling_aligned[i]
        
        # Mean reversion signals
        price_below_vwap = close[i] < vwap[i]
        price_above_vwap = close[i] > vwap[i]
        at_lower_band = low[i] <= vwap_lower[i]
        at_upper_band = high[i] >= vwap_upper[i]
        
        if position == 0:
            if uptrend and price_below_vwap and at_lower_band:
                # Uptrend: buy dip to VWAP lower band
                signals[i] = 0.25
                position = 1
            elif downtrend and price_above_vwap and at_upper_band:
                # Downtrend: sell rally to VWAP upper band
                signals[i] = -0.25
                position = -1
        else:
            # Exit conditions
            if position == 1:
                # Exit long: price reaches VWAP or breaks above upper band
                exit_signal = (close[i] >= vwap[i]) or (high[i] >= vwap_upper[i] * 1.05)
                if exit_signal:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = 0.25
            elif position == -1:
                # Exit short: price reaches VWAP or breaks below lower band
                exit_signal = (close[i] <= vwap[i]) or (low[i] <= vwap_lower[i] * 0.95)
                if exit_signal:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = -0.25
    
    return signals
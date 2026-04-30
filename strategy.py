#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Donchian(20) breakout with volume confirmation (>2.0x 20-bar avg volume) and choppiness regime filter (CHOP > 61.8 = range → mean reversion, CHOP < 38.2 = trend → trend follow).
# Uses Donchian channel for structural breakouts, volume spike to confirm institutional interest, and choppiness index to adapt to market regime.
# In trending regimes (CHOP < 38.2): trade breakout direction. In ranging regimes (CHOP > 61.8): trade mean reversion at Donchian extremes.
# ATR trailing stop (2.5x) for risk management. Discrete position sizing at ±0.25 to limit fee drag.
# Session filter (08:00-20:00 UTC) to avoid low-liquidity periods. Target: 75-150 total trades over 4 years.

name = "4h_Donchian20_VolumeSpike_ChopRegime_ATRStop_v1"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Pre-compute session hours (08-20 UTC) to avoid look-ahead
    hours = pd.DatetimeIndex(prices["open_time"]).hour
    in_session = (hours >= 8) & (hours <= 20)
    
    # ATR(14) for volatility and stoploss
    atr_period = 14
    tr1 = high[1:] - low[1:]
    tr2 = np.abs(high[1:] - close[:-1])
    tr3 = np.abs(low[1:] - close[:-1])
    tr = np.concatenate([[np.max([tr1[0], tr2[0], tr3[0]])], np.maximum(tr1, np.maximum(tr2, tr3))])
    atr = pd.Series(tr).rolling(window=atr_period, min_periods=atr_period).mean().values
    
    # Volume confirmation: volume > 2.0x 20-period average (tighter to reduce trades)
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_confirm = volume > (2.0 * vol_ma_20)
    
    # Donchian(20) on 4h data
    donchian_period = 20
    donchian_upper = pd.Series(high).rolling(window=donchian_period, min_periods=donchian_period).max().values
    donchian_lower = pd.Series(low).rolling(window=donchian_period, min_periods=donchian_period).min().values
    
    # Choppiness Index (CHOP) - uses high/low/close to measure trend vs range
    chop_period = 14
    # True Range
    tr_chop = np.maximum(high - low, np.maximum(np.abs(high - np.roll(close, 1)), np.abs(low - np.roll(close, 1))))
    tr_chop[0] = high[0] - low[0]  # first period TR
    # Sum of TR over chop_period
    sum_tr = pd.Series(tr_chop).rolling(window=chop_period, min_periods=chop_period).sum().values
    # Highest high and lowest low over chop_period
    hh = pd.Series(high).rolling(window=chop_period, min_periods=chop_period).max().values
    ll = pd.Series(low).rolling(window=chop_period, min_periods=chop_period).min().values
    # Avoid division by zero
    range_hl = hh - ll
    range_hl = np.where(range_hl == 0, 1e-10, range_hl)
    # Choppiness Index formula: 100 * log10(sum(tr) / (n * log10(range))) / log10(n)
    chop = 100 * np.log10(sum_tr / (chop_period * np.log10(range_hl))) / np.log10(chop_period)
    # Handle invalid values
    chop = np.where(np.isnan(chop) | np.isinf(chop), 50.0, chop)
    
    # Regime filters: CHOP > 61.8 = ranging (mean revert), CHOP < 38.2 = trending (trend follow)
    chop_range = chop > 61.8   # ranging market → mean reversion
    chop_trend = chop < 38.2   # trending market → trend following
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    highest_since_entry = 0.0
    lowest_since_entry = 0.0
    
    start_idx = max(20, 14, 20) + 10  # warmup for Donchian, ATR, CHOP
    
    for i in range(start_idx, n):
        # Skip if indicators not available or outside session
        if (np.isnan(donchian_upper[i]) or 
            np.isnan(donchian_lower[i]) or
            np.isnan(atr[i]) or
            np.isnan(volume_confirm[i]) or
            np.isnan(chop[i]) or
            not in_session[i]):
            signals[i] = 0.0
            continue
        
        curr_close = close[i]
        curr_high = high[i]
        curr_low = low[i]
        curr_donchian_upper = donchian_upper[i]
        curr_donchian_lower = donchian_lower[i]
        curr_atr = atr[i]
        curr_volume_confirm = volume_confirm[i]
        curr_chop = chop[i]
        curr_chop_range = chop_range[i]
        curr_chop_trend = chop_trend[i]
        
        if position == 0:  # Flat - look for new entries
            # In trending regime: trade breakout direction
            if curr_chop_trend:
                # Long: price breaks above Donchian upper with volume confirmation
                if curr_close > curr_donchian_upper and curr_volume_confirm:
                    signals[i] = 0.25
                    position = 1
                    entry_price = curr_close
                    highest_since_entry = curr_close
                # Short: price breaks below Donchian lower with volume confirmation
                elif curr_close < curr_donchian_lower and curr_volume_confirm:
                    signals[i] = -0.25
                    position = -1
                    entry_price = curr_close
                    lowest_since_entry = curr_close
            # In ranging regime: trade mean reversion at extremes
            elif curr_chop_range:
                # Long: price touches or breaks below Donchian lower (oversold) with volume confirmation
                if curr_close <= curr_donchian_lower and curr_volume_confirm:
                    signals[i] = 0.25
                    position = 1
                    entry_price = curr_close
                    highest_since_entry = curr_close
                # Short: price touches or breaks above Donchian upper (overbought) with volume confirmation
                elif curr_close >= curr_donchian_upper and curr_volume_confirm:
                    signals[i] = -0.25
                    position = -1
                    entry_price = curr_close
                    lowest_since_entry = curr_close
        
        elif position == 1:  # Long position
            # Update highest price since entry
            highest_since_entry = max(highest_since_entry, curr_high)
            # ATR trailing stop: exit if price drops 2.5*ATR from highest point
            if curr_close < highest_since_entry - (2.5 * curr_atr):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:  # Short position
            # Update lowest price since entry
            lowest_since_entry = min(lowest_since_entry, curr_low)
            # ATR trailing stop: exit if price rises 2.5*ATR from lowest point
            if curr_close > lowest_since_entry + (2.5 * curr_atr):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals
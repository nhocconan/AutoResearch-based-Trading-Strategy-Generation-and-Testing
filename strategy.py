#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Choppiness Index regime filter + Donchian(20) breakout + volume confirmation
# Uses weekly Choppiness Index to distinguish trending (CHOP < 38.2) vs ranging (CHOP > 61.8) markets.
# In trending markets: trade Donchian breakouts with volume confirmation (>1.5x 20-period avg volume).
# In ranging markets: fade moves at Bollinger Bands (20,2) with RSI(14) extremes.
# Weekly regime filter reduces whipsaws in sideways markets, improving performance in both bull and bear.
# Target: 20-40 trades/year to minimize fee decay while capturing strong directional moves.

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get weekly data for Choppiness Index calculation
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 14:
        return np.zeros(n)
    
    # Calculate weekly Choppiness Index (14-period)
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    n_1w = len(close_1w)
    
    # True Range
    tr = np.zeros(n_1w)
    for i in range(n_1w):
        if i == 0:
            tr[i] = high_1w[i] - low_1w[i]
        else:
            hl = high_1w[i] - low_1w[i]
            hc = abs(high_1w[i] - close_1w[i-1])
            lc = abs(low_1w[i] - close_1w[i-1])
            tr[i] = max(hl, hc, lc)
    
    # ATR (14-period smoothed)
    atr_1w = np.zeros(n_1w)
    for i in range(n_1w):
        if i < 13:
            atr_1w[i] = np.nan
        elif i == 13:
            atr_1w[i] = np.mean(tr[0:14])
        else:
            atr_1w[i] = (atr_1w[i-1] * 13 + tr[i]) / 14
    
    # Choppiness Index = 100 * log10(sum(ATR)/ (n * (max(high)-min(low)))) / log10(n)
    chop_1w = np.full(n_1w, np.nan)
    lookback = 14
    for i in range(lookback-1, n_1w):
        if np.isnan(atr_1w[i]):
            chop_1w[i] = np.nan
            continue
        sum_atr = np.sum(atr_1w[i-lookback+1:i+1])
        max_high = np.max(high_1w[i-lookback+1:i+1])
        min_low = np.min(low_1w[i-lookback+1:i+1])
        if max_high == min_low:
            chop_1w[i] = 50.0  # neutral when no range
        else:
            chop_1w[i] = 100 * np.log10(sum_atr / (lookback * (max_high - min_low))) / np.log10(lookback)
    
    # Align weekly Choppiness Index to 4h
    chop_1w_aligned = align_htf_to_ltf(prices, df_1w, chop_1w)
    
    # Donchian channel (20-period) on 4h
    dc_period = 20
    upper_dc = np.full(n, np.nan)
    lower_dc = np.full(n, np.nan)
    
    for i in range(dc_period, n):
        upper_dc[i] = np.max(high[i-dc_period:i])
        lower_dc[i] = np.min(low[i-dc_period:i])
    
    # Bollinger Bands (20,2) for ranging markets
    bb_period = 20
    bb_std = 2
    sma = np.full(n, np.nan)
    std_dev = np.full(n, np.nan)
    upper_bb = np.full(n, np.nan)
    lower_bb = np.full(n, np.nan)
    
    for i in range(bb_period, n):
        sma[i] = np.mean(close[i-bb_period:i])
        std_dev[i] = np.std(close[i-bb_period:i])
        upper_bb[i] = sma[i] + bb_std * std_dev[i]
        lower_bb[i] = sma[i] - bb_std * std_dev[i]
    
    # RSI (14) for mean reversion signals
    rsi_period = 14
    rsi = np.full(n, np.nan)
    delta = np.diff(close, prepend=close[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    
    avg_gain = np.zeros(n)
    avg_loss = np.zeros(n)
    for i in range(n):
        if i < rsi_period:
            avg_gain[i] = np.nan
            avg_loss[i] = np.nan
        elif i == rsi_period:
            avg_gain[i] = np.mean(gain[0:rsi_period])
            avg_loss[i] = np.mean(loss[0:rsi_period])
        else:
            avg_gain[i] = (avg_gain[i-1] * (rsi_period-1) + gain[i]) / rsi_period
            avg_loss[i] = (avg_loss[i-1] * (rsi_period-1) + loss[i]) / rsi_period
    
    rs = np.where(avg_loss != 0, avg_gain / avg_loss, np.inf)
    rsi = 100 - (100 / (1 + rs))
    
    # Volume confirmation: 1.5x 20-period average volume
    vol_period = 20
    vol_ma = np.full(n, np.nan)
    for i in range(vol_period, n):
        vol_ma[i] = np.mean(volume[i-vol_period:i])
    
    signals = np.zeros(n)
    position = 0
    size = 0.25  # 25% position size
    
    # Warmup period
    start_idx = max(dc_period, bb_period, rsi_period, vol_period)
    
    for i in range(start_idx, n):
        if (np.isnan(chop_1w_aligned[i]) or 
            np.isnan(upper_dc[i]) or 
            np.isnan(lower_dc[i]) or
            np.isnan(upper_bb[i]) or
            np.isnan(lower_bb[i]) or
            np.isnan(rsi[i]) or
            np.isnan(vol_ma[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        vol_ratio = volume[i] / vol_ma[i] if vol_ma[i] > 0 else 0
        
        # Regime determination
        is_trending = chop_1w_aligned[i] < 38.2
        is_ranging = chop_1w_aligned[i] > 61.8
        
        if position == 0:
            if is_trending:
                # Trending market: Donchian breakout with volume confirmation
                breakout_up = price > upper_dc[i]
                breakout_down = price < lower_dc[i]
                volume_confirm = vol_ratio > 1.5
                
                if breakout_up and volume_confirm:
                    signals[i] = size
                    position = 1
                elif breakout_down and volume_confirm:
                    signals[i] = -size
                    position = -1
                else:
                    signals[i] = 0.0
            elif is_ranging:
                # Ranging market: mean reversion at Bollinger Bands with RSI extremes
                at_upper_bb = price >= upper_bb[i]
                at_lower_bb = price <= lower_bb[i]
                rsi_overbought = rsi[i] > 70
                rsi_oversold = rsi[i] < 30
                
                if at_lower_bb and rsi_oversold:
                    signals[i] = size
                    position = 1
                elif at_upper_bb and rsi_overbought:
                    signals[i] = -size
                    position = -1
                else:
                    signals[i] = 0.0
            else:
                # Transition zone: no trade
                signals[i] = 0.0
        elif position == 1:
            # Long exit conditions
            if is_trending:
                # Exit on Donchian breakdown or loss of trend
                if price < lower_dc[i] or chop_1w_aligned[i] > 61.8:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = size
            else:
                # Exit ranging position at middle or opposite band
                if price >= sma[i] or price >= upper_bb[i]:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = size
        elif position == -1:
            # Short exit conditions
            if is_trending:
                # Exit on Donchian breakout or loss of trend
                if price > upper_dc[i] or chop_1w_aligned[i] > 61.8:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = -size
            else:
                # Exit ranging position at middle or opposite band
                if price <= sma[i] or price <= lower_bb[i]:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = -size
    
    return signals

name = "4h_ChoppinessRegime_DonchianBB_RSI"
timeframe = "4h"
leverage = 1.0
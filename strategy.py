#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Hypothesis: 4h Camarilla pivot breakout with 1d volume spike and chop regime filter
    # Long: Close > H3 pivot AND volume > 1.5x 20-period average AND chop < 61.8 (trending)
    # Short: Close < L3 pivot AND volume > 1.5x 20-period average AND chop < 61.8 (trending)
    # Exit: Close crosses back to H4/L4 or chop > 61.8 (range) or volume dry-up
    # Using 4h timeframe for optimal trade frequency, Camarilla for institutional levels,
    # volume for confirmation, chop to avoid false breakouts in ranging markets.
    # Discrete position sizing (0.25) to minimize fee churn.
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1d data for Camarilla pivot calculation
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    # Calculate Camarilla pivot levels from prior 1d bar
    # H4 = close + 1.5*(high-low)
    # H3 = close + 1.25*(high-low)
    # H2 = close + 1.083*(high-low)
    # H1 = close + 0.833*(high-low)
    # Pivot = (high+low+close)/3
    # L1 = close - 0.833*(high-low)
    # L2 = close - 1.083*(high-low)
    # L3 = close - 1.25*(high-low)
    # L4 = close - 1.5*(high-low)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Use previous day's data for today's pivot levels (avoid look-ahead)
    high_1d_lag = np.concatenate([[np.nan], high_1d[:-1]])
    low_1d_lag = np.concatenate([[np.nan], low_1d[:-1]])
    close_1d_lag = np.concatenate([[np.nan], close_1d[:-1]])
    
    range_1d = high_1d_lag - low_1d_lag
    pivot = (high_1d_lag + low_1d_lag + close_1d_lag) / 3
    
    H4 = close_1d_lag + 1.5 * range_1d
    H3 = close_1d_lag + 1.25 * range_1d
    L3 = close_1d_lag - 1.25 * range_1d
    L4 = close_1d_lag - 1.5 * range_1d
    
    # Align 1d Camarilla levels to 4h
    H3_4h = align_htf_to_ltf(prices, df_1d, H3)
    L3_4h = align_htf_to_ltf(prices, df_1d, L3)
    H4_4h = align_htf_to_ltf(prices, df_1d, H4)
    L4_4h = align_htf_to_ltf(prices, df_1d, L4)
    
    # Get 1d data for chop regime filter
    # Chop = 100 * log10(sum(ATR(1), n) / (log10(n) * (max(high,n) - min(low,n))))
    # Simplified: chop > 61.8 = ranging, chop < 38.2 = trending
    # We'll use a simpler version: chop = 100 * |close - ema| / atr
    
    # Calculate 1d ATR(14) for chop calculation
    tr_1d = np.maximum(
        high_1d_lag[1:] - low_1d_lag[1:],
        np.maximum(
            np.abs(high_1d_lag[1:] - close_1d_lag[:-1]),
            np.abs(low_1d_lag[1:] - close_1d_lag[:-1])
        )
    )
    tr_1d = np.concatenate([[np.nan], tr_1d])
    
    atr_1d = np.full(len(tr_1d), np.nan)
    for i in range(14, len(tr_1d)):
        if i == 14:
            atr_1d[i] = np.nanmean(tr_1d[1:15])
        else:
            atr_1d[i] = (atr_1d[i-1] * 13 + tr_1d[i]) / 14
    
    # Calculate 1d EMA(20) for chop calculation
    ema_1d = np.full(len(close_1d_lag), np.nan)
    multiplier = 2 / (20 + 1)
    for i in range(len(close_1d_lag)):
        if i == 0:
            ema_1d[i] = close_1d_lag[i]
        elif not np.isnan(ema_1d[i-1]) and not np.isnan(close_1d_lag[i]):
            ema_1d[i] = (close_1d_lag[i] - ema_1d[i-1]) * multiplier + ema_1d[i-1]
    
    # Calculate chop: 100 * |close - ema| / atr
    chop_1d = np.full(len(close_1d_lag), np.nan)
    for i in range(len(chop_1d)):
        if not np.isnan(ema_1d[i]) and not np.isnan(atr_1d[i]) and atr_1d[i] > 0:
            chop_1d[i] = 100 * np.abs(close_1d_lag[i] - ema_1d[i]) / atr_1d[i]
    
    # Align chop to 4h
    chop_4h = align_htf_to_ltf(prices, df_1d, chop_1d)
    
    # Get 1d volume for confirmation (>1.5x 20-period average)
    vol_ma_1d = np.full(len(volume), np.nan)
    for i in range(20, len(volume)):
        vol_ma_1d[i] = np.mean(volume[i-20:i])
    volume_spike_1d = volume > (1.5 * vol_ma_1d)
    
    # Align volume spike to 4h
    volume_spike_4h = align_htf_to_ltf(prices, df_1d, volume_spike_1d)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(50, n):
        # Skip if data not ready
        if (np.isnan(H3_4h[i]) or np.isnan(L3_4h[i]) or 
            np.isnan(chop_4h[i]) or np.isnan(volume_spike_4h[i])):
            signals[i] = 0.0
            continue
        
        # Regime filter: chop < 61.8 = trending (good for breakouts)
        trending_regime = chop_4h[i] < 61.8
        
        # Volume confirmation
        vol_confirm = volume_spike_4h[i]
        
        # Entry logic: Camarilla breakout + volume + regime
        long_entry = (close[i] > H3_4h[i]) and vol_confirm and trending_regime
        short_entry = (close[i] < L3_4h[i]) and vol_confirm and trending_regime
        
        # Exit logic: reverse breakout or regime change or volume dry-up
        long_exit = (close[i] < H4_4h[i]) or (not trending_regime) or (not vol_confirm)
        short_exit = (close[i] > L4_4h[i]) or (not trending_regime) or (not vol_confirm)
        
        if long_entry and position != 1:
            position = 1
            signals[i] = 0.25
        elif short_entry and position != -1:
            position = -1
            signals[i] = -0.25
        elif position == 1 and long_exit:
            position = 0
            signals[i] = 0.0
        elif position == -1 and short_exit:
            position = 0
            signals[i] = 0.0
        else:
            # Hold current position
            if position == 1:
                signals[i] = 0.25
            elif position == -1:
                signals[i] = -0.25
            else:
                signals[i] = 0.0
    
    return signals

name = "4h_1d_camarilla_breakout_volume_chop_v1"
timeframe = "4h"
leverage = 1.0
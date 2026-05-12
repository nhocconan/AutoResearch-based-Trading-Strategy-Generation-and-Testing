#!/usr/bin/env python3
# 1d_PV_CRSI_Range_Reversal
# Hypothesis: On 1d timeframe, use PV_CRSI (price-volume adjusted RSI) to identify extreme mean-reversion opportunities
# in ranging markets. Enter long when PV_CRSI < 15 and price below VWAP, short when PV_CRSI > 85 and price above VWAP.
# Use weekly ADX < 20 as range filter to avoid trending markets. Exit when PV_CRSI returns to neutral (40-60 range).
# Targets 15-25 trades/year to minimize fee drift while capturing mean reversion in BTC/ETH ranging regimes.

name = "1d_PV_CRSI_Range_Reversal"
timeframe = "1d"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Calculate VWAP (typical price * volume cumulative)
    typical_price = (high + low + close) / 3.0
    vwap_num = np.cumsum(typical_price * volume)
    vwap_den = np.cumsum(volume)
    vwap = vwap_num / vwap_den
    
    # Calculate PV_CRSI (Price-Volume Weighted RSI)
    # 1. Price change
    delta = np.diff(close, prepend=close[0])
    # 2. Volume-weighted gains/losses
    vol_weighted_gain = np.where(delta > 0, delta * volume, 0.0)
    vol_weighted_loss = np.where(delta < 0, -delta * volume, 0.0)
    # 3. Smoothed averages (Wilder's smoothing)
    alpha = 1.0 / 14
    avg_gain = np.zeros_like(vol_weighted_gain)
    avg_loss = np.zeros_like(vol_weighted_loss)
    avg_gain[0] = vol_weighted_gain[0]
    avg_loss[0] = vol_weighted_loss[0]
    for i in range(1, n):
        avg_gain[i] = (1 - alpha) * avg_gain[i-1] + alpha * vol_weighted_gain[i]
        avg_loss[i] = (1 - alpha) * avg_loss[i-1] + alpha * vol_weighted_loss[i]
    # 4. RSI calculation
    rs = np.divide(avg_gain, avg_loss, out=np.zeros_like(avg_gain), where=avg_loss!=0)
    pvcrsi = 100 - (100 / (1 + rs))
    
    # Load weekly data for ADX range filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 30:
        return np.zeros(n)
    
    # Calculate ADX on weekly data
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    
    # True Range
    tr1 = high_1w - low_1w
    tr2 = np.abs(high_1w - np.concatenate([[close_1w[0]], close_1w[:-1]]))
    tr3 = np.abs(low_1w - np.concatenate([[close_1w[0]], close_1w[:-1]]))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    
    # Directional Movement
    dm_plus = np.where((high_1w - np.concatenate([[high_1w[0]], high_1w[:-1]])) > 
                       (np.concatenate([[low_1w[0]], low_1w[:-1]]) - low_1w), 
                       np.maximum(high_1w - np.concatenate([[high_1w[0]], high_1w[:-1]]), 0), 0)
    dm_minus = np.where((np.concatenate([[low_1w[0]], low_1w[:-1]]) - low_1w) > 
                        (high_1w - np.concatenate([[high_1w[0]], high_1w[:-1]])), 
                        np.maximum(np.concatenate([[low_1w[0]], low_1w[:-1]]) - low_1w, 0), 0)
    
    # Smoothed values
    def wilders_smoothing(data, period):
        result = np.zeros_like(data)
        result[period-1] = np.nansum(data[:period])
        for i in range(period, len(data)):
            result[i] = result[i-1] - (result[i-1] / period) + data[i]
        return result
    
    tr14 = wilders_smoothing(tr, 14)
    dm_plus_14 = wilders_smoothing(dm_plus, 14)
    dm_minus_14 = wilders_smoothing(dm_minus, 14)
    
    # DI and DX
    di_plus = np.divide(dm_plus_14, tr14, out=np.zeros_like(dm_plus_14), where=tr14!=0) * 100
    di_minus = np.divide(dm_minus_14, tr14, out=np.zeros_like(dm_minus_14), where=tr14!=0) * 100
    dx = np.divide(np.abs(di_plus - di_minus), (di_plus + di_minus), out=np.zeros_like(di_plus), where=(di_plus + di_minus)!=0) * 100
    adx = wilders_smoothing(dx, 14)
    
    # Align weekly ADX to daily (wait for completed weekly bar)
    adx_aligned = align_htf_to_ltf(prices, df_1w, adx)
    
    # Volume confirmation: current volume > 1.3 * 20-day average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_confirm = volume > (1.3 * vol_ma)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 50  # Ensure indicators are stable
    
    for i in range(start_idx, n):
        # Skip if any critical data is not ready
        if (np.isnan(pvcrsi[i]) or np.isnan(vwap[i]) or 
            np.isnan(adx_aligned[i]) or np.isnan(vol_ma[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue
        
        pvcrsi_val = pvcrsi[i]
        vwap_val = vwap[i]
        adx_val = adx_aligned[i]
        vol_confirm = volume_confirm[i]
        
        if position == 0:
            # LONG: PV_CRSI oversold (<15), price below VWAP, ranging market (ADX<20), volume confirmation
            if pvcrsi_val < 15 and close[i] < vwap_val and adx_val < 20 and vol_confirm:
                signals[i] = 0.25
                position = 1
            # SHORT: PV_CRSI overbought (>85), price above VWAP, ranging market (ADX<20), volume confirmation
            elif pvcrsi_val > 85 and close[i] > vwap_val and adx_val < 20 and vol_confirm:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: PV_CRSI returns to neutral range (40-60) or breaks above VWAP with strength
            if pvcrsi_val > 40 and pvcrsi_val < 60:
                signals[i] = 0.0
                position = 0
            elif close[i] > vwap_val and pvcrsi_val > 50:  # Early exit if momentum shifts
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: PV_CRSI returns to neutral range (40-60) or breaks below VWAP with weakness
            if pvcrsi_val > 40 and pvcrsi_val < 60:
                signals[i] = 0.0
                position = 0
            elif close[i] < vwap_val and pvcrsi_val < 50:  # Early exit if momentum shifts
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals
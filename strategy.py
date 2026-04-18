#!/usr/bin/env python3
"""
4h_ADX_VWAP_Retest
Hypothesis: In trending markets (ADX>25), price pulls back to VWAP before continuing.
Entries occur on pullbacks to VWAP with volume confirmation (>1.5x average).
Exit when trend weakens (ADX<20) or price extends too far (>2*ATR from VWAP).
Designed for low trade frequency (15-25/year) on 4h timeframe to minimize fee drag.
Works in both bull and bear by following the trend via ADX filter.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Calculate VWAP (typical price * volume) cumulative
    typical_price = (high + low + close) / 3.0
    tpv = typical_price * volume
    cum_tpv = np.cumsum(tpv)
    cum_vol = np.cumsum(volume)
    vwap = np.where(cum_vol > 0, cum_tpv / cum_vol, 0.0)
    
    # Calculate ATR for stop loss and extension filter
    tr1 = high[1:] - low[1:]
    tr2 = np.abs(high[1:] - close[:-1])
    tr3 = np.abs(low[1:] - close[:-1])
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr = np.concatenate([[np.nan], tr])
    
    atr_period = 14
    atr = np.full_like(tr, np.nan)
    if len(tr) >= atr_period:
        atr[atr_period] = np.nanmean(tr[1:atr_period+1])
        for i in range(atr_period+1, len(tr)):
            atr[i] = (atr[i-1] * (atr_period-1) + tr[i]) / atr_period
    
    # Calculate ADX for trend strength
    # Directional Movement
    dm_plus = np.where((high[1:] - high[:-1]) > (low[:-1] - low[1:]), 
                       np.maximum(high[1:] - high[:-1], 0), 0)
    dm_minus = np.where((low[:-1] - low[1:]) > (high[1:] - high[:-1]), 
                        np.maximum(low[:-1] - low[1:], 0), 0)
    dm_plus = np.concatenate([[np.nan], dm_plus])
    dm_minus = np.concatenate([[np.nan], dm_minus])
    
    # Smooth TR, DM+ and DM-
    tr_smooth = np.full_like(tr, np.nan)
    dm_plus_smooth = np.full_like(dm_plus, np.nan)
    dm_minus_smooth = np.full_like(dm_minus, np.nan)
    
    if len(tr) >= atr_period:
        tr_smooth[atr_period] = np.nanmean(tr[1:atr_period+1])
        dm_plus_smooth[atr_period] = np.nanmean(dm_plus[1:atr_period+1])
        dm_minus_smooth[atr_period] = np.nanmean(dm_minus[1:atr_period+1])
        
        for i in range(atr_period+1, len(tr)):
            tr_smooth[i] = (tr_smooth[i-1] * (atr_period-1) + tr[i]) / atr_period
            dm_plus_smooth[i] = (dm_plus_smooth[i-1] * (atr_period-1) + dm_plus[i]) / atr_period
            dm_minus_smooth[i] = (dm_minus_smooth[i-1] * (atr_period-1) + dm_minus[i]) / atr_period
    
    # DI+ and DI-
    di_plus = np.full_like(dm_plus_smooth, np.nan)
    di_minus = np.full_like(dm_minus_smooth, np.nan)
    valid = ~np.isnan(tr_smooth) & (tr_smooth != 0)
    di_plus[valid] = 100 * dm_plus_smooth[valid] / tr_smooth[valid]
    di_minus[valid] = 100 * dm_minus_smooth[valid] / tr_smooth[valid]
    
    # DX and ADX
    dx = np.full_like(di_plus, np.nan)
    dx_valid = ~np.isnan(di_plus) & ~np.isnan(di_minus) & ((di_plus + di_minus) != 0)
    dx[dx_valid] = 100 * np.abs(di_plus[dx_valid] - di_minus[dx_valid]) / (di_plus[dx_valid] + di_minus[dx_valid])
    
    adx = np.full_like(dx, np.nan)
    if len(dx) >= atr_period:
        adx[2*atr_period-1] = np.nanmean(dx[atr_period:2*atr_period])
        for i in range(2*atr_period, len(dx)):
            adx[i] = (adx[i-1] * (atr_period-1) + dx[i]) / atr_period
    
    # Volume confirmation: volume > 1.5x 20-period average
    vol_ma = np.full_like(volume, np.nan)
    vol_period = 20
    if len(volume) >= vol_period:
        for i in range(vol_period, len(volume)):
            vol_ma[i] = np.mean(volume[i - vol_period:i])
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(atr_period*2, vol_period) + 1
    
    for i in range(start_idx, n):
        # Skip if any required data is not available
        if (np.isnan(adx[i]) or np.isnan(vwap[i]) or np.isnan(atr[i]) or 
            np.isnan(vol_ma[i])):
            signals[i] = 0.0
            continue
        
        # Volume confirmation
        vol_confirm = volume[i] > 1.5 * vol_ma[i]
        
        # Distance from VWAP in ATR units
        if atr[i] > 0:
            dist_from_vwap = abs(close[i] - vwap[i]) / atr[i]
        else:
            dist_from_vwap = 0
        
        if position == 0:
            # Long: ADX > 25 (trending), price near VWAP (<1 ATR), volume confirmation
            if adx[i] > 25 and dist_from_vwap < 1.0 and vol_confirm:
                signals[i] = 0.25
                position = 1
            # Short: ADX > 25 (trending), price near VWAP (<1 ATR), volume confirmation
            elif adx[i] > 25 and dist_from_vwap < 1.0 and vol_confirm:
                # For short, we need price to be below VWAP in downtrend
                # We'll use price < VWAP as additional filter for short
                if close[i] < vwap[i]:
                    signals[i] = -0.25
                    position = -1
        
        elif position == 1:
            # Long exit: trend weakens (ADX<20) or price extends too far (>2*ATR from VWAP)
            if adx[i] < 20 or dist_from_vwap > 2.0:
                signals[i] = -0.25  # reverse to short
                position = -1
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: trend weakens (ADX<20) or price extends too far (>2*ATR from VWAP)
            if adx[i] < 20 or dist_from_vwap > 2.0:
                signals[i] = 0.25  # reverse to long
                position = 1
            else:
                signals[i] = -0.25
    
    return signals

name = "4h_ADX_VWAP_Retest"
timeframe = "4h"
leverage = 1.0
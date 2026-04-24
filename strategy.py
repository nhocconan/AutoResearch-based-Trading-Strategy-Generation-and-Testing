#!/usr/bin/env python3
"""
Hypothesis: 6h Elder Ray Index (Bull/Bear Power) with 1w ADX regime filter and volume spike confirmation.
- Primary timeframe: 6h targeting 50-150 total trades over 4 years (12-37/year).
- HTF: 1w ADX(14) for regime filter (ADX > 25 = trending, ADX < 20 = ranging).
- Entry: Long when Bull Power > 0 and Bear Power < 0 in trending regime with volume > 1.5 * 6h volume MA(30);
         Short when Bear Power < 0 and Bull Power > 0 in trending regime with volume > 1.5 * 6h volume MA(30).
- Exit: Opposite Elder Ray signal (Long exits when Bear Power > 0, Short exits when Bull Power < 0).
- Signal size: 0.25 discrete to balance capture and fee control.
- Elder Ray measures bull/bear strength via EMA(13); ADX regime avoids whipsaws in low-volatility environments;
  volume spike confirms institutional participation. Works in bull (buying strength) and bear (selling weakness).
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Extract price data
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1w data for ADX calculation
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 14:
        return np.zeros(n)
    
    # Calculate 1w ADX(14)
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    
    # True Range
    tr1 = np.abs(high_1w[1:] - low_1w[1:])
    tr2 = np.abs(high_1w[1:] - close_1w[:-1])
    tr3 = np.abs(low_1w[1:] - close_1w[:-1])
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr = np.concatenate([[np.abs(high_1w[0] - low_1w[0])], tr])
    
    # Directional Movement
    dm_plus = np.where((high_1w[1:] - high_1w[:-1]) > (low_1w[:-1] - low_1w[1:]),
                       np.maximum(high_1w[1:] - high_1w[:-1], 0), 0)
    dm_minus = np.where((low_1w[:-1] - low_1w[1:]) > (high_1w[1:] - high_1w[:-1]),
                        np.maximum(low_1w[:-1] - low_1w[1:], 0), 0)
    dm_plus = np.concatenate([[0], dm_plus])
    dm_minus = np.concatenate([[0], dm_minus])
    
    # Smoothed TR, DM+
    tr_period = 14
    tr_sum = np.zeros_like(tr)
    dm_plus_sum = np.zeros_like(dm_plus)
    dm_minus_sum = np.zeros_like(dm_minus)
    
    # Initial values
    tr_sum[tr_period-1] = tr[:tr_period].sum()
    dm_plus_sum[tr_period-1] = dm_plus[:tr_period].sum()
    dm_minus_sum[tr_period-1] = dm_minus[:tr_period].sum()
    
    # Wilder's smoothing
    for i in range(tr_period, len(tr)):
        tr_sum[i] = tr_sum[i-1] - (tr_sum[i-1] / tr_period) + tr[i]
        dm_plus_sum[i] = dm_plus_sum[i-1] - (dm_plus_sum[i-1] / tr_period) + dm_plus[i]
        dm_minus_sum[i] = dm_minus_sum[i-1] - (dm_minus_sum[i-1] / tr_period) + dm_minus[i]
    
    # DI+ and DI-
    di_plus = 100 * dm_plus_sum / tr_sum
    di_minus = 100 * dm_minus_sum / tr_sum
    
    # DX and ADX
    dx = np.abs(di_plus - di_minus) / (di_plus + di_minus) * 100
    dx = np.where(np.isnan(dx), 0, dx)
    
    adx = np.zeros_like(dx)
    adx[2*tr_period-1] = dx[tr_period:2*tr_period].mean()
    for i in range(2*tr_period, len(dx)):
        adx[i] = (adx[i-1] * (tr_period-1) + dx[i]) / tr_period
    
    adx_aligned = align_htf_to_ltf(prices, df_1w, adx)
    
    # Get 6h data for volume MA and EMA(13) for Elder Ray
    df_6h = get_htf_data(prices, '6h')
    if len(df_6h) < 30:
        return np.zeros(n)
    
    # Calculate 6h EMA(13) for Elder Ray
    close_6h = df_6h['close'].values
    ema_13 = pd.Series(close_6h).ewm(span=13, adjust=False, min_periods=13).mean().values
    ema_13_aligned = align_htf_to_ltf(prices, df_6h, ema_13)
    
    # Calculate 6h volume MA(30) for confirmation
    volume_6h = df_6h['volume'].values
    vol_ma_6h = pd.Series(volume_6h).rolling(window=30, min_periods=30).mean().values
    vol_ma_6h_aligned = align_htf_to_ltf(prices, df_6h, vol_ma_6h)
    
    # Elder Ray: Bull Power = High - EMA(13), Bear Power = Low - EMA(13)
    bull_power = high - ema_13
    bear_power = low - ema_13
    
    # Align Elder Ray components
    bull_power_aligned = align_htf_to_ltf(prices, df_6h, bull_power)
    bear_power_aligned = align_htf_to_ltf(prices, df_6h, bear_power)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start from index where all indicators are ready
    start_idx = max(30, 13, 30)  # EMA needs 13, volume MA needs 30, ADX needs ~28
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(adx_aligned[i]) or np.isnan(ema_13_aligned[i]) or 
            np.isnan(vol_ma_6h_aligned[i]) or np.isnan(bull_power_aligned[i]) or 
            np.isnan(bear_power_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        curr_close = close[i]
        curr_high = high[i]
        curr_low = low[i]
        curr_volume = volume[i]
        curr_bull_power = bull_power_aligned[i]
        curr_bear_power = bear_power_aligned[i]
        
        # Regime filter: ADX > 25 = trending (good for Elder Ray), ADX < 20 = ranging (avoid)
        trending_regime = adx_aligned[i] > 25
        ranging_regime = adx_aligned[i] < 20
        
        # Volume confirmation: 1.5x threshold (balanced to avoid overtrading)
        vol_confirm = curr_volume > 1.5 * vol_ma_6h_aligned[i]
        
        if position == 0:
            # Check for entry signals in trending regime only
            if trending_regime and vol_confirm:
                # Long: Bull Power > 0 (strong buying pressure) and Bear Power < 0 (weak selling pressure)
                if curr_bull_power > 0 and curr_bear_power < 0:
                    signals[i] = 0.25
                    position = 1
                # Short: Bear Power < 0 (strong selling pressure) and Bull Power > 0 (weak buying pressure)
                elif curr_bear_power < 0 and curr_bull_power > 0:
                    signals[i] = -0.25
                    position = -1
        elif position == 1:
            # Long position: exit when Bear Power becomes positive (selling pressure overwhelms)
            if curr_bear_power > 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short position: exit when Bull Power becomes positive (buying pressure overwhelms)
            if curr_bull_power > 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "6h_ElderRay_1wADX_Regime_VolumeConfirm_v1"
timeframe = "6h"
leverage = 1.0
#!/usr/bin/env python3
"""
6h_WeeklyPivot_PriceChannelBreakout_VolumeSpike_HT
Hypothesis: Breakouts from weekly pivot-based price channels (CPR) with volume confirmation and 1d EMA trend filter capture strong trending moves while avoiding false breakouts. Weekly Central Pivot Range (CPR) acts as dynamic support/resistance. Works in bull/bear by following 1d trend direction.
"""

name = "6h_WeeklyPivot_PriceChannelBreakout_VolumeSpike_HT"
timeframe = "6h"
leverage = 1.0

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

    # Get weekly data ONCE before loop
    df_wk = get_htf_data(prices, '1w')

    # Calculate Weekly Central Pivot Range (CPR)
    # TC = (H + L + C) / 3
    # BC = (H + L) / 2
    # TC > BC: TC is pivot, BC is base
    # TC < BC: BC is pivot, TC is base
    wk_high = df_wk['high'].values
    wk_low = df_wk['low'].values
    wk_close = df_wk['close'].values

    tc = (wk_high + wk_low + wk_close) / 3  # True Pivot
    bc = (wk_high + wk_low) / 2             # Base Pivot

    pivot = np.where(tc >= bc, tc, bc)
    base = np.where(tc >= bc, bc, tc)

    # Shift by 1 to use previous week's data
    prev_pivot = np.roll(pivot, 1)
    prev_base = np.roll(base, 1)
    prev_pivot[0] = np.nan
    prev_base[0] = np.nan

    # Weekly CPR boundaries (support and resistance)
    wk_cpr_top = np.maximum(prev_pivot, prev_base)
    wk_cpr_bottom = np.minimum(prev_pivot, prev_base)

    # Align weekly CPR to 6h timeframe
    wk_cpr_top_aligned = align_htf_to_ltf(prices, df_wk, wk_cpr_top)
    wk_cpr_bottom_aligned = align_htf_to_ltf(prices, df_wk, wk_cpr_bottom)

    # 1d EMA34 trend filter
    df_1d = get_htf_data(prices, '1d')
    close_1d = df_1d['close'].values
    ema_34_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)

    # Volume spike: >1.5x 20-period average (6h)
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (1.5 * vol_ma)

    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short

    for i in range(34, n):  # Start after EMA34 warmup
        if (np.isnan(wk_cpr_top_aligned[i]) or np.isnan(wk_cpr_bottom_aligned[i]) or 
            np.isnan(ema_34_1d_aligned[i]) or np.isnan(volume_spike[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue

        if position == 0:
            # LONG: Price breaks above weekly CPR top + 1d EMA34 uptrend + volume spike
            if (close[i] > wk_cpr_top_aligned[i] and 
                close[i] > ema_34_1d_aligned[i] and 
                volume_spike[i]):
                signals[i] = 0.25
                position = 1
            # SHORT: Price breaks below weekly CPR bottom + 1d EMA34 downtrend + volume spike
            elif (close[i] < wk_cpr_bottom_aligned[i] and 
                  close[i] < ema_34_1d_aligned[i] and 
                  volume_spike[i]):
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: Price closes below weekly CPR bottom (reversal)
            if close[i] < wk_cpr_bottom_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: Price closes above weekly CPR top (reversal)
            if close[i] > wk_cpr_top_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25

    return signals
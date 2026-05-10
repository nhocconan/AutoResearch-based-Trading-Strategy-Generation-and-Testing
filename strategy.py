# 1d_WeeklyPivot_Breakout_1wTrend_Volume
# Hypothesis: Price breaks above/below weekly Camarilla Pivot (R1/S1) with weekly EMA50 trend filter and daily volume confirmation.
# Works in bull/bear by trading in direction of weekly trend. Targets 15-25 trades/year (60-100 total) to minimize fee drag.
# Uses weekly data for structure, daily data for execution timing and volume confirmation.
# Weekly Camarilla levels provide strong support/resistance; breakouts with volume and trend alignment capture directional moves.

name = "1d_WeeklyPivot_Breakout_1wTrend_Volume"
timeframe = "1d"
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
    
    # Weekly data
    df_w = get_htf_data(prices, '1w')
    high_w = df_w['high'].values
    low_w = df_w['low'].values
    close_w = df_w['close'].values
    volume_w = df_w['volume'].values
    
    # Weekly Camarilla levels from prior week: R1 = close + 1.1*(high-low)/12, S1 = close - 1.1*(high-low)/12
    camarilla_r1_w = close_w + 1.1 * (high_w - low_w) / 12
    camarilla_s1_w = close_w - 1.1 * (high_w - low_w) / 12
    
    # Weekly EMA50 for trend filter
    ema50_w = np.full(len(close_w), np.nan)
    if len(close_w) >= 50:
        ema50_w[49] = np.mean(close_w[:50])
        alpha = 2 / (50 + 1)
        for i in range(50, len(close_w)):
            ema50_w[i] = alpha * close_w[i] + (1 - alpha) * ema50_w[i-1]
    
    # Daily volume SMA20 for volume confirmation
    vol_sma20_d = np.full(len(prices), np.nan)
    if len(prices) >= 20:
        vol_sma20_d[19] = np.mean(volume[:20])
        for i in range(20, len(prices)):
            vol_sma20_d[i] = (vol_sma20_d[i-1] * 19 + volume[i]) / 20
    
    # Align weekly indicators to daily
    r1_w_aligned = align_htf_to_ltf(prices, df_w, camarilla_r1_w)
    s1_w_aligned = align_htf_to_ltf(prices, df_w, camarilla_s1_w)
    ema50_w_aligned = align_htf_to_ltf(prices, df_w, ema50_w)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 50  # Wait for weekly EMA50
    
    for i in range(start_idx, n):
        if np.isnan(r1_w_aligned[i]) or np.isnan(s1_w_aligned[i]) or np.isnan(ema50_w_aligned[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Volume confirmation: current daily volume > 1.5x average daily volume
        volume_confirm = volume[i] > 1.5 * vol_sma20_d[i]
        
        # Trend and price relative to weekly Camarilla levels
        is_uptrend = close[i] > ema50_w_aligned[i]
        is_downtrend = close[i] < ema50_w_aligned[i]
        price_above_r1 = close[i] > r1_w_aligned[i]
        price_below_s1 = close[i] < s1_w_aligned[i]
        
        if position == 0:
            # Long: price breaks above weekly R1, in weekly uptrend, with volume
            if price_above_r1 and is_uptrend and volume_confirm:
                signals[i] = 0.25
                position = 1
            # Short: price breaks below weekly S1, in weekly downtrend, with volume
            elif price_below_s1 and is_downtrend and volume_confirm:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit: price falls back below weekly R1 or weekly trend turns down
            if not price_above_r1 or not is_uptrend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit: price rises back above weekly S1 or weekly trend turns up
            if not price_below_s1 or not is_downtrend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals
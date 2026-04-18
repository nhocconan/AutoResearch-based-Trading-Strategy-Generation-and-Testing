# 1d_WeeklyCamarilla_H3L3_Breakout_1wTrendFilter_V1
# Hypothesis: 1d Camarilla H3/L3 breakout with 1w trend filter and volume confirmation.
# Works in bull (breakouts with 1w trend) and bear (mean reversion at H3/L3 in range) via price action at key levels.
# Target: 25-75 trades/year (100-300 total over 4 years) to avoid fee drag while capturing significant moves.
# Uses 1w trend to filter direction and avoid counter-trend trades in strong trends.

#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "1d_WeeklyCamarilla_H3L3_Breakout_1wTrendFilter_V1"
timeframe = "1d"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Load weekly data for trend filter
    df_1w = get_htf_data(prices, '1w')
    
    # Calculate weekly EMA34 for trend filter (slow EMA for weekly)
    close_1w = df_1w['close'].values
    ema_34_1w = pd.Series(close_1w).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_34_1w)
    
    # Load daily data for Camarilla pivot levels and volume average
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate Camarilla pivot levels (H3, L3) from previous daily bar
    prev_close = df_1d['close'].shift(1).values
    prev_high = df_1d['high'].shift(1).values
    prev_low = df_1d['low'].shift(1).values
    
    pivot = (prev_high + prev_low + prev_close) / 3
    range_hl = prev_high - prev_low
    H3 = pivot + (range_hl * 1.1 / 4)
    L3 = pivot - (range_hl * 1.1 / 4)
    
    # Align daily H3/L3 to daily (wait for daily close)
    H3_aligned = align_htf_to_ltf(prices, df_1d, H3)
    L3_aligned = align_htf_to_ltf(prices, df_1d, L3)
    
    # Volume filter: current volume > 1.3 * 20-day average
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_filter = volume > (1.3 * vol_ma_20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 100  # Wait for indicator calculations
    
    for i in range(start_idx, n):
        # Skip if any required data is not available
        if (np.isnan(H3_aligned[i]) or np.isnan(L3_aligned[i]) or
            np.isnan(ema_34_1w_aligned[i]) or np.isnan(vol_ma_20[i])):
            signals[i] = 0.0
            continue
        
        close_val = close[i]
        H3_val = H3_aligned[i]
        L3_val = L3_aligned[i]
        ema_34_1w_val = ema_34_1w_aligned[i]
        vol_filter = volume_filter[i]
        
        if position == 0:
            # Long: break above H3 with volume confirmation and above weekly EMA34 (uptrend)
            if close_val > H3_val and vol_filter and (close_val > ema_34_1w_val):
                signals[i] = 0.25
                position = 1
            # Short: break below L3 with volume confirmation and below weekly EMA34 (downtrend)
            elif close_val < L3_val and vol_filter and (close_val < ema_34_1w_val):
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Long exit: price falls below L3 or weekly trend turns down
            if close_val < L3_val or (close_val < ema_34_1w_val):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: price rises above H3 or weekly trend turns up
            if close_val > H3_val or (close_val > ema_34_1w_val):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals
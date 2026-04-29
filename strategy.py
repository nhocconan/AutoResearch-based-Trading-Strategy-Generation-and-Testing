#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h WilliamsVix Fix mean reversion + daily trend filter + volume confirmation
# WilliamsVix Fix identifies exhaustion points in any market condition
# Daily EMA34 provides trend regime (avoid counter-trend trades in strong trends)
# Volume confirmation ensures mean reversion has follow-through
# Works in bull/bear/range by fading extremes only when aligned with higher timeframe trend
# Target: 15-25 trades/year (60-100 total over 4 years)

name = "6h_WilliamsVixFix_DailyTrend_VolumeConfirm_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Load HTF data ONCE before loop for daily calculations
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # Calculate daily EMA34 for trend filter
    close_1d = pd.Series(df_1d['close'].values)
    ema34_1d = close_1d.ewm(span=34, adjust=False, min_periods=34).mean().values
    ema34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema34_1d)
    
    # Calculate WilliamsVix Fix (22-period) on 6h data
    # WVF = ((Highest High - Lowest Low) / Highest High) * 100
    # where Highest High = highest high in lookback period
    # and Lowest Low = lowest low in lookback period
    highest_high = pd.Series(high).rolling(window=22, min_periods=22).max().values
    lowest_low = pd.Series(low).rolling(window=22, min_periods=22).min().values
    # Avoid division by zero
    hh_ll_diff = highest_high - lowest_low
    wvf = np.where(hh_ll_diff != 0, (hh_ll_diff / highest_high) * 100, 0)
    
    # Volume confirmation: volume > 1.5x 20-period average
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_confirm = volume > (1.5 * vol_ma_20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(50, 22, 20, 34)  # warmup for all indicators
    
    for i in range(start_idx, n):
        # Skip if HTF data not available
        if np.isnan(ema34_1d_aligned[i]) or np.isnan(wvf[i]):
            signals[i] = 0.0
            continue
            
        curr_close = close[i]
        curr_wvf = wvf[i]
        curr_volume_confirm = volume_confirm[i]
        curr_ema34_1d = ema34_1d_aligned[i]
        
        # Determine trend regime from daily EMA34
        # Bullish: price > daily EMA34
        # Bearish: price < daily EMA34
        
        if position == 0:  # Flat - look for new entries
            # WilliamsVix Fix < 30 indicates exhaustion (mean reversion setup)
            if curr_wvf < 30 and curr_volume_confirm:
                # In bullish regime: look for long setups
                if curr_close > curr_ema34_1d:
                    signals[i] = 0.25
                    position = 1
                # In bearish regime: look for short setups
                elif curr_close < curr_ema34_1d:
                    signals[i] = -0.25
                    position = -1
                # In range/choppy regime (price near EMA): no new entries
        
        elif position == 1:  # Long position - exit conditions
            # Exit when: WVF > 50 (exhaustion gone) OR price crosses below daily EMA34
            if curr_wvf > 50 or curr_close < curr_ema34_1d:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:  # Short position - exit conditions
            # Exit when: WVF > 50 (exhaustion gone) OR price crosses above daily EMA34
            if curr_wvf > 50 or curr_close > curr_ema34_1d:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals
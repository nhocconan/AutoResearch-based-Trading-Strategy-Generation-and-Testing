#!/usr/bin/env python3
# 4h_VWAP_Deviation_ZScore_MeanReversion_v1
# Hypothesis: Mean-reversion using VWAP deviation Z-score with 1-day trend filter and volume confirmation.
# Works in bull/bear by fading extremes only when higher timeframe trend aligns.
# Targets 20-35 trades/year via strict Z-score > 2.0 or < -2.0 requirement.

name = "4h_VWAP_Deviation_ZScore_MeanReversion_v1"
timeframe = "4h"
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
    open_price = prices['open'].values
    volume = prices['volume'].values
    
    # === 1d Data (loaded ONCE) ===
    df_1d = get_htf_data(prices, '1d')
    close_1d = df_1d['close'].values
    
    # === VWAP Deviation Calculation ===
    typical_price = (high + low + close) / 3
    tp_vol = typical_price * volume
    cum_tp_vol = np.cumsum(tp_vol)
    cum_vol = np.cumsum(volume)
    vwap = cum_tp_vol / cum_vol
    vwap_dev = (close - vwap) / vwap  # percentage deviation
    
    # Rolling Z-score of VWAP deviation (20-period)
    vwap_dev_series = pd.Series(vwap_dev)
    vwap_zscore = (vwap_dev_series - vwap_dev_series.rolling(20, min_periods=20).mean()) / \
                  vwap_dev_series.rolling(20, min_periods=20).std(ddof=0)
    vwap_zscore = vwap_zscore.fillna(0).values
    
    # === 1d EMA50 Trend Filter ===
    ema50_1d = pd.Series(close_1d).ewm(span=50, min_periods=50, adjust=False).mean().values
    ema50_1d_4h = align_htf_to_ltf(prices, df_1d, ema50_1d)
    
    # === Volume Spike Filter ===
    vol_ema20 = pd.Series(volume).ewm(span=20, min_periods=20, adjust=False).mean().values
    volume_ok = volume > vol_ema20 * 1.5
    
    # === Signal Parameters ===
    position_size = 0.25  # 25% of capital per trade
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    holding_bars = 0
    
    # Start after warmup
    start_idx = 50
    
    for i in range(start_idx, n):
        # Skip if any required data is invalid
        if (np.isnan(vwap_zscore[i]) or np.isnan(ema50_1d_4h[i]) or 
            np.isnan(volume_ok[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
                holding_bars = 0
            else:
                signals[i] = 0.0
            continue
        
        if position == 0:
            # Long: VWAP deviation < -2.0 (undervalued) + above 1d EMA50 + volume spike
            if (vwap_zscore[i] < -2.0 and 
                close[i] > ema50_1d_4h[i] and 
                volume_ok[i]):
                signals[i] = position_size
                position = 1
                holding_bars = 0
            # Short: VWAP deviation > 2.0 (overvalued) + below 1d EMA50 + volume spike
            elif (vwap_zscore[i] > 2.0 and 
                  close[i] < ema50_1d_4h[i] and 
                  volume_ok[i]):
                signals[i] = -position_size
                position = -1
                holding_bars = 0
        else:
            # Enforce minimum holding period (8 bars)
            holding_bars += 1
            if holding_bars < 8:
                signals[i] = position_size if position == 1 else -position_size
                continue
            
            # Exit: VWAP deviation returns to neutral (|Z| < 0.5) or opposite extreme
            if position == 1:
                if vwap_zscore[i] > -0.5 or vwap_zscore[i] > 1.0:
                    signals[i] = 0.0
                    position = 0
                    holding_bars = 0
                else:
                    signals[i] = position_size
            elif position == -1:
                if vwap_zscore[i] < 0.5 or vwap_zscore[i] < -1.0:
                    signals[i] = 0.0
                    position = 0
                    holding_bars = 0
                else:
                    signals[i] = -position_size
    
    return signals
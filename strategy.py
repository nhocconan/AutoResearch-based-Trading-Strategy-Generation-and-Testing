#!/usr/bin/env python3
"""
1h_OrderBook_Imbalance_4hTrend_Volume
Hypothesis: Order book imbalance (proxy via tick rule + volume) predicts short-term direction. 
Trades only when aligned with 4h EMA20 trend and confirmed by volume spikes. 
Uses 4h for trend direction and 1h for entry timing to avoid overtrading. 
Designed for low turnover (~20-30 trades/year) to minimize fee drag in ranging 2025 markets.
"""

name = "1h_OrderBook_Imbalance_4hTrend_Volume"
timeframe = "1h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    volume = prices['volume'].values
    taker_buy_volume = prices['taker_buy_volume'].values
    
    # === Tick Rule: Buy Volume Proxy ===
    # Use taker buy volume as proxy for aggressive buying pressure
    # When taker buy volume > 50% of total volume, bias is bullish
    volume_imbalance = taker_buy_volume - (volume - taker_buy_volume)  # buy - sell
    volume_imbalance_ratio = volume_imbalance / volume  # normalized [-1, 1]
    
    # === 4h EMA20 Trend Filter ===
    df_4h = get_htf_data(prices, '4h')
    close_4h = df_4h['close'].values
    ema20_4h = pd.Series(close_4h).ewm(span=20, min_periods=20, adjust=False).mean().values
    ema20_4h_aligned = align_htf_to_ltf(prices, df_4h, ema20_4h)
    
    # === Volume Spike Filter (20-period EMA) ===
    vol_ema20 = pd.Series(volume).ewm(span=20, min_periods=20, adjust=False).mean().values
    volume_ok = volume > vol_ema20 * 1.5  # Require 1.5x average volume
    
    # === Session Filter: 08:00-20:00 UTC ===
    # Pre-compute hours from DatetimeIndex (already datetime64[ms])
    hours = prices.index.hour
    session_ok = (hours >= 8) & (hours <= 20)
    
    # === Signal Parameters ===
    position_size = 0.20
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after warmup
    start_idx = 30  # covers EMA20
    
    for i in range(start_idx, n):
        # Skip if any required data is invalid
        if (np.isnan(ema20_4h_aligned[i]) or np.isnan(volume_ok[i]) or 
            np.isnan(volume_imbalance_ratio[i])):
            if position == 1:
                signals[i] = 0.0
            elif position == -1:
                signals[i] = 0.0
            else:
                signals[i] = 0.0
            continue
        
        if not session_ok[i]:
            if position == 1:
                signals[i] = 0.0
            elif position == -1:
                signals[i] = 0.0
            else:
                signals[i] = 0.0
            continue
        
        if position == 0:
            # Long: Positive imbalance + above 4h EMA20 + volume spike
            if volume_imbalance_ratio[i] > 0.15 and close[i] > ema20_4h_aligned[i] and volume_ok[i]:
                signals[i] = position_size
                position = 1
            # Short: Negative imbalance + below 4h EMA20 + volume spike
            elif volume_imbalance_ratio[i] < -0.15 and close[i] < ema20_4h_aligned[i] and volume_ok[i]:
                signals[i] = -position_size
                position = -1
        else:
            # Exit conditions: reversal of imbalance or trend
            if position == 1:
                # Exit: Negative imbalance or price below 4h EMA20
                if volume_imbalance_ratio[i] < -0.05 or close[i] < ema20_4h_aligned[i]:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = position_size
            elif position == -1:
                # Exit: Positive imbalance or price above 4h EMA20
                if volume_imbalance_ratio[i] > 0.05 or close[i] > ema20_4h_aligned[i]:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = -position_size
    
    return signals
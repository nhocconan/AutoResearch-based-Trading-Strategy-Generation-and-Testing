#!/usr/bin/env python3
"""
Hypothesis: 1h Camarilla H3/L3 breakout with 4h EMA50 trend filter and daily volume spike confirmation.
- Camarilla pivot levels (H3/L3) on 1h chart using prior bar's OHLC act as intraday support/resistance
- Breakout above H3 with daily volume > 2.0x 20-day average signals strong momentum
- Breakdown below L3 with daily volume > 2.0x 20-day average signals strong bearish momentum
- 4h EMA50 ensures trades align with higher timeframe trend (avoid counter-trend)
- Session filter (08-20 UTC) reduces noise during low-liquidity hours
- Discrete position size 0.20 to minimize drawdown during crashes like 2022
- Target: 15-30 trades/year on 1h timeframe (60-120 total over 4 years)
- Works in both bull/bear via 4h trend filter and volatility-adjusted breakouts
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    open_time = prices['open_time'].values
    
    # Calculate typical price for Camarilla pivots (using prior bar's OHLC)
    typical_price = (high + low + close) / 3.0
    
    # Shift by 1 to use prior bar's data for pivot calculation (no look-ahead)
    typical_price_shifted = np.roll(typical_price, 1)
    high_shifted = np.roll(high, 1)
    low_shifted = np.roll(low, 1)
    close_shifted = np.roll(close, 1)
    
    # Set first bar to NaN since we don't have prior bar data
    typical_price_shifted[0] = np.nan
    high_shifted[0] = np.nan
    low_shifted[0] = np.nan
    close_shifted[0] = np.nan
    
    # Camarilla pivot levels (based on prior bar)
    pivot = (high_shifted + low_shifted + close_shifted) / 3.0
    range_hl = high_shifted - low_shifted
    
    # Resistance/support levels (H3/L3 = 1.1*(H-L)/6 from pivot)
    H3 = pivot + (range_hl * 1.1 / 6.0)  # H3 = pivot + 1.1*(H-L)/6
    L3 = pivot - (range_hl * 1.1 / 6.0)  # L3 = pivot - 1.1*(H-L)/6
    
    # 4h data for EMA50 trend filter
    df_4h = get_htf_data(prices, '4h')
    close_4h = df_4h['close'].values
    ema_50_4h = pd.Series(close_4h).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_4h_aligned = align_htf_to_ltf(prices, df_4h, ema_50_4h)
    
    # 1d data for volume spike confirmation (> 2.0x 20-day average)
    df_1d = get_htf_data(prices, '1d')
    volume_1d = df_1d['volume'].values
    vol_ma_20_1d = pd.Series(volume_1d).rolling(window=20, min_periods=20).mean().values
    vol_ma_20_1d_aligned = align_htf_to_ltf(prices, df_1d, vol_ma_20_1d)
    
    # Session filter: 08-20 UTC (pre-compute hours array)
    hours = pd.DatetimeIndex(open_time).hour
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start from index where all indicators are ready
    start_idx = max(50, 20)  # 4h EMA50, 1d volume MA20
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(H3[i]) or np.isnan(L3[i]) or np.isnan(ema_50_4h_aligned[i]) or 
            np.isnan(vol_ma_20_1d_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Session filter: only trade 08-20 UTC
        hour = hours[i]
        in_session = 8 <= hour <= 20
        
        # Volume confirmation: current 1d volume > 2.0x 20-day average
        volume_confirm = volume_1d[i] > 2.0 * vol_ma_20_1d_aligned[i]
        
        if position == 0 and in_session:
            # Long: Close > H3 AND price above 4h EMA50 AND volume confirmation AND in session
            if close[i] > H3[i] and close[i] > ema_50_4h_aligned[i] and volume_confirm:
                signals[i] = 0.20
                position = 1
            # Short: Close < L3 AND price below 4h EMA50 AND volume confirmation AND in session
            elif close[i] < L3[i] and close[i] < ema_50_4h_aligned[i] and volume_confirm:
                signals[i] = -0.20
                position = -1
        elif position == 1:
            # Long exit: Close < pivot OR price crosses below 4h EMA50 OR outside session
            if close[i] < pivot[i] or close[i] < ema_50_4h_aligned[i] or not in_session:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.20
        elif position == -1:
            # Short exit: Close > pivot OR price crosses above 4h EMA50 OR outside session
            if close[i] > pivot[i] or close[i] > ema_50_4h_aligned[i] or not in_session:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.20
    
    return signals

name = "1h_Camarilla_H3L3_Breakout_4hEMA50_1dVolumeSpike_v1"
timeframe = "1h"
leverage = 1.0
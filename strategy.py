#!/usr/bin/env python3
"""
Experiment #3774: 1h Donchian(20) breakout + 4h EMA(50) trend filter + 1d volume spike
HYPOTHESIS: 1h Donchian breakouts capture short-term swings, filtered by 4h EMA(50) for trend direction and 1d volume spike for institutional participation. Works in bull markets (breakouts above EMA50 with volume) and bear markets (breakdowns below EMA50 with volume). Position size 0.20 manages drawdown. Target: 60-150 total trades over 4 years = 15-37/year for 1h.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "exp_3774_1h_donchian20_4h_ema50_1d_vol_v1"
timeframe = "1h"
leverage = 1.0

def generate_signals(prices):
    close = prices["close"].values.astype(np.float64)
    high = prices["high"].values.astype(np.float64)
    low = prices["low"].values.astype(np.float64)
    volume = prices["volume"].values.astype(np.float64)
    n = len(close)
    
    # === HTF: 4h data for EMA(50) trend filter (Call ONCE before loop) ===
    df_4h = get_htf_data(prices, '4h')
    close_4h = df_4h['close'].values
    ema_4h = pd.Series(close_4h).ewm(span=50, min_periods=50, adjust=False).mean().values
    ema_4h_aligned = align_htf_to_ltf(prices, df_4h, ema_4h)
    
    # === HTF: 1d data for volume spike detection (Call ONCE before loop) ===
    df_1d = get_htf_data(prices, '1d')
    volume_1d = df_1d['volume'].values
    vol_ma_1d = pd.Series(volume_1d).rolling(window=20, min_periods=20).mean().values
    vol_spike_1d = np.ones(len(volume_1d))
    vol_spike_1d[20:] = (volume_1d[20:] / vol_ma_1d[20:]) > 2.0  # 2x average volume
    vol_spike_1d_aligned = align_htf_to_ltf(prices, df_1d, vol_spike_1d)
    
    # === 1h Indicators: Donchian Channel(20) for breakout ===
    lookback_dc = 20
    highest_high = pd.Series(high).rolling(window=lookback_dc, min_periods=lookback_dc).max().values
    lowest_low = pd.Series(low).rolling(window=lookback_dc, min_periods=lookback_dc).min().values
    
    # === 1h Indicators: ATR(14) for stoploss ===
    tr1 = high[1:] - low[1:]
    tr2 = np.abs(high[1:] - close[:-1])
    tr3 = np.abs(low[1:] - close[:-1])
    tr = np.concatenate([[np.nan], np.maximum(tr1, np.maximum(tr2, tr3))])
    atr = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    # === Signals Initialization ===
    signals = np.zeros(n)
    SIZE = 0.20  # 20% position size
    
    # Position tracking state variables
    in_position = False
    position_side = 0
    entry_price = 0.0
    
    warmup = max(lookback_dc, 50, 20, 14)  # sufficient for all indicators
    
    for i in range(warmup, n):
        # --- Data Validity Check ---
        if (np.isnan(highest_high[i]) or np.isnan(lowest_low[i]) or
            np.isnan(ema_4h_aligned[i]) or np.isnan(vol_spike_1d_aligned[i]) or
            np.isnan(atr[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        
        # --- Exit Logic: Stoploss at 2.5*ATR ---
        if in_position:
            if position_side > 0:  # Long
                if price < entry_price - 2.5 * atr[i]:
                    in_position = False
                    position_side = 0
                    signals[i] = 0.0
                else:
                    signals[i] = SIZE
            else:  # Short
                if price > entry_price + 2.5 * atr[i]:
                    in_position = False
                    position_side = 0
                    signals[i] = 0.0
                else:
                    signals[i] = -SIZE
            continue
        
        # --- Session Filter: 08-20 UTC ---
        hour = pd.Timestamp(prices.iloc[i]["open_time"]).hour
        if hour < 8 or hour > 20:
            signals[i] = 0.0
            continue
        
        # --- New Position Entry Logic ---
        # Require volume spike AND Donchian breakout aligned with 4h EMA trend
        if vol_spike_1d_aligned[i]:
            # Long entry: Price breaks above Donchian upper band AND above 4h EMA50 (bullish)
            if price > highest_high[i-1] and price > ema_4h_aligned[i]:
                in_position = True
                position_side = 1
                entry_price = close[i]
                signals[i] = SIZE
            # Short entry: Price breaks below Donchian lower band AND below 4h EMA50 (bearish)
            elif price < lowest_low[i-1] and price < ema_4h_aligned[i]:
                in_position = True
                position_side = -1
                entry_price = close[i]
                signals[i] = -SIZE
            else:
                signals[i] = 0.0
        else:
            signals[i] = 0.0
    
    return signals
#!/usr/bin/env python3
"""
Experiment #250: 1d Donchian(20) breakout + 1w HMA trend + volume confirmation
HYPOTHESIS: Daily Donchian breakouts capture medium-term trends. Weekly HMA (21) filters 
direction - only trade breakouts aligned with weekly trend. Volume confirmation (>2x average) 
ensures institutional participation. ATR(14) stoploss (2.5x) manages risk. Discrete sizing 0.25.
Works in bull markets via upward breakouts and bear markets via downward breakouts. 
Target: 50-100 total trades over 4 years (12-25/year). Avoids overtrading with strict 3-condition entry.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "exp_250_1d_donchian20_1w_hma_vol_v1"
timeframe = "1d"
leverage = 1.0

def generate_signals(prices):
    close = prices["close"].values.astype(np.float64)
    high = prices["high"].values.astype(np.float64)
    low = prices["low"].values.astype(np.float64)
    volume = prices["volume"].values.astype(np.float64)
    n = len(close)
    
    # === HTF: 1w data for HMA(21) trend filter ===
    df_1w = get_htf_data(prices, '1w')
    close_1w = df_1w['close'].values
    
    # Hull Moving Average (HMA) calculation
    def hma(series, period):
        half_period = period // 2
        sqrt_period = int(np.sqrt(period))
        wma_half = pd.Series(series).ewm(span=half_period, adjust=False).mean()
        wma_full = pd.Series(series).ewm(span=period, adjust=False).mean()
        hma_raw = 2 * wma_half - wma_full
        hma_result = pd.Series(hma_raw).ewm(span=sqrt_period, adjust=False).mean()
        return hma_result.values
    
    hma_21_1w = hma(close_1w, 21)
    hma_21_1w_aligned = align_htf_to_ltf(prices, df_1w, hma_21_1w)
    
    # === 1d Indicators: Donchian(20) channels ===
    def donchian_channels(high, low, period):
        upper = pd.Series(high).rolling(window=period, min_periods=period).max().values
        lower = pd.Series(low).rolling(window=period, min_periods=period).min().values
        return upper, lower
    
    donch_20_upper, donch_20_lower = donchian_channels(high, low, 20)
    
    # === 1d Indicators: ATR(14) for stoploss and volatility ===
    tr = np.zeros(n)
    tr[0] = high[0] - low[0]
    for i in range(1, n):
        tr[i] = max(high[i] - low[i], abs(high[i] - close[i-1]), abs(low[i] - close[i-1]))
    atr_14 = pd.Series(tr).ewm(span=14, min_periods=14, adjust=False).mean().values
    
    # === 1d Indicators: Volume confirmation ===
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_ratio = np.zeros(n)
    vol_ratio[20:] = volume[20:] / vol_ma_20[20:]
    vol_ratio[:20] = 1.0
    
    # === Signals Initialization ===
    signals = np.zeros(n)
    SIZE = 0.25
    
    # Position tracking
    in_position = False
    position_side = 0
    entry_price = 0.0
    bars_since_entry = 0
    
    warmup = 60  # Enough for 20-period Donchian and 20-period volume MA
    
    for i in range(warmup, n):
        # --- Data validity check ---
        if (np.isnan(donch_20_upper[i]) or np.isnan(donch_20_lower[i]) or
            np.isnan(atr_14[i]) or np.isnan(vol_ratio[i]) or
            np.isnan(hma_21_1w_aligned[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        
        # --- Volume confirmation: require significant volume spike ---
        volume_confirm = vol_ratio[i] > 2.0
        
        # --- Weekly trend filter: price relative to weekly HMA ---
        weekly_uptrend = price > hma_21_1w_aligned[i]
        weekly_downtrend = price < hma_21_1w_aligned[i]
        
        # --- Donchian breakout conditions ---
        breakout_up = high[i] > donch_20_upper[i-1]  # Break above upper channel
        breakout_down = low[i] < donch_20_lower[i-1]  # Break below lower channel
        
        # --- Exit logic: ATR-based stoploss ---
        if in_position:
            bars_since_entry += 1
            
            if position_side > 0:  # Long position
                stop_level = entry_price - 2.5 * atr_14[i]
                if low[i] < stop_level:
                    in_position = False
                    position_side = 0
                    bars_since_entry = 0
                    signals[i] = 0.0
                    continue
                # Exit on opposite breakout with volume
                if breakout_down and volume_confirm:
                    in_position = False
                    position_side = 0
                    bars_since_entry = 0
                    signals[i] = 0.0
                    continue
            else:  # Short position
                stop_level = entry_price + 2.5 * atr_14[i]
                if high[i] > stop_level:
                    in_position = False
                    position_side = 0
                    bars_since_entry = 0
                    signals[i] = 0.0
                    continue
                # Exit on opposite breakout with volume
                if breakout_up and volume_confirm:
                    in_position = False
                    position_side = 0
                    bars_since_entry = 0
                    signals[i] = 0.0
                    continue
            
            # Maintain position
            signals[i] = position_side * SIZE
            continue
        
        # --- Entry logic: Donchian breakout with weekly trend and volume ---
        if volume_confirm:
            # Long entry: upward breakout in weekly uptrend
            if breakout_up and weekly_uptrend:
                in_position = True
                position_side = 1
                entry_price = close[i]
                bars_since_entry = 0
                signals[i] = SIZE
            # Short entry: downward breakout in weekly downtrend
            elif breakout_down and weekly_downtrend:
                in_position = True
                position_side = -1
                entry_price = close[i]
                bars_since_entry = 0
                signals[i] = -SIZE
            else:
                signals[i] = 0.0
        else:
            signals[i] = 0.0
    
    return signals
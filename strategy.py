#!/usr/bin/env python3
"""
Experiment #5615: 6h Donchian(20) breakout + weekly Camarilla pivot + volume confirmation
HYPOTHESIS: On 6h timeframe, Donchian(20) breakouts with volume > 2.0x average and aligned with 
weekly Camarilla pivot levels (R3/S3 for mean reversion, R4/S4 for breakout) capture high-probability 
trades. Weekly pivot provides structural support/resistance from higher timeframe (1w), reducing 
false breakouts in ranging markets and enhancing continuation in trending markets. Works in both 
bull (breakouts with weekly R4 support) and bear (breakdowns with weekly S4 resistance). 
ATR-based trailing stop (2.0x ATR) limits drawdown. Discrete position sizing (0.25) minimizes fee churn. 
Target: 12-37 trades/year (50-150 total over 4 years).
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "exp_5615_6h_donchian20_1w_camarilla_vol_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    close = prices["close"].values.astype(np.float64)
    high = prices["high"].values.astype(np.float64)
    low = prices["low"].values.astype(np.float64)
    volume = prices["volume"].values.astype(np.float64)
    n = len(close)
    
    # Precompute session hours once (open_time is already datetime64[ms])
    hours = pd.DatetimeIndex(prices["open_time"]).hour
    
    # === HTF: 1w data for Camarilla pivot levels ===
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) >= 1:
        # Calculate weekly Camarilla pivot levels
        high_1w = pd.Series(df_1w['high'].values)
        low_1w = pd.Series(df_1w['low'].values)
        close_1w = pd.Series(df_1w['close'].values)
        
        # Pivot point (PP)
        pp = (high_1w + low_1w + close_1w) / 3
        
        # Camarilla levels
        r4 = pp + (high_1w - low_1w) * 1.1 / 2
        r3 = pp + (high_1w - low_1w) * 1.1 / 4
        s3 = pp - (high_1w - low_1w) * 1.1 / 4
        s4 = pp - (high_1w - low_1w) * 1.1 / 2
        
        # Align to LTF (6h)
        r4_aligned = align_htf_to_ltf(prices, df_1w, r4.values)
        r3_aligned = align_htf_to_ltf(prices, df_1w, r3.values)
        s3_aligned = align_htf_to_ltf(prices, df_1w, s3.values)
        s4_aligned = align_htf_to_ltf(prices, df_1w, s4.values)
    else:
        r4_aligned = np.full(n, np.nan)
        r3_aligned = np.full(n, np.nan)
        s3_aligned = np.full(n, np.nan)
        s4_aligned = np.full(n, np.nan)
    
    # === 6h Indicators: Donchian Channel (20-period) ===
    donchian_high = pd.Series(high).rolling(window=20, min_periods=20).max().values
    donchian_low = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # === 6h Indicators: Volume confirmation ===
    avg_volume = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_ratio = volume / np.where(avg_volume > 0, avg_volume, 1)
    
    # === 6h Indicators: ATR(14) for trailing stop ===
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]
    atr = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    # === Signals Initialization ===
    signals = np.zeros(n)
    SIZE = 0.25  # 25% position size (discrete level)
    
    # Position tracking state variables
    in_position = False
    position_side = 0
    entry_price = 0.0
    highest_since_entry = 0.0
    lowest_since_entry = 0.0
    
    warmup = max(20, 20, 14)  # Donchian, volume avg, ATR
    
    for i in range(warmup, n):
        # --- Session Filter: Avoid low liquidity periods ---
        hour = hours[i]
        if 21 <= hour <= 23:
            signals[i] = 0.0
            continue
        
        # --- Data Validity Check ---
        if (np.isnan(donchian_high[i]) or np.isnan(donchian_low[i]) or 
            np.isnan(volume_ratio[i]) or np.isnan(atr[i]) or
            np.isnan(r4_aligned[i]) or np.isnan(r3_aligned[i]) or
            np.isnan(s3_aligned[i]) or np.isnan(s4_aligned[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        
        # --- Exit Logic ---
        if in_position:
            if position_side > 0:  # Long position
                highest_since_entry = max(highest_since_entry, high[i])
                stop_price = highest_since_entry - 2.0 * atr[i]
                # Exit: stoploss OR price breaks below Donchian low (trend reversal)
                if price <= stop_price or price <= donchian_low[i]:
                    in_position = False
                    position_side = 0
                    signals[i] = 0.0
                else:
                    signals[i] = SIZE
            else:  # Short position
                lowest_since_entry = min(lowest_since_entry, low[i])
                stop_price = lowest_since_entry + 2.0 * atr[i]
                # Exit: stoploss OR price breaks above Donchian high (trend reversal)
                if price >= stop_price or price >= donchian_high[i]:
                    in_position = False
                    position_side = 0
                    signals[i] = 0.0
                else:
                    signals[i] = -SIZE
            continue
        
        # --- New Position Entry Logic ---
        breakout_up = price > donchian_high[i-1]
        breakout_down = price < donchian_low[i-1]
        volume_confirmed = volume_ratio[i] > 2.0
        
        # Camarilla pivot logic:
        # Long: breakout above Donchian high with price > weekly R3 (mean reversion zone) OR 
        #       breakout above weekly R4 (breakout continuation)
        # Short: breakout below Donchian low with price < weekly S3 (mean reversion zone) OR 
        #        breakout below weekly S4 (breakout continuation)
        long_setup = breakout_up and volume_confirmed and (
            (price > r3_aligned[i]) or (price > r4_aligned[i])
        )
        short_setup = breakout_down and volume_confirmed and (
            (price < s3_aligned[i]) or (price < s4_aligned[i])
        )
        
        if long_setup:
            in_position = True
            position_side = 1
            entry_price = close[i]
            highest_since_entry = high[i]
            lowest_since_entry = low[i]
            signals[i] = SIZE
        elif short_setup:
            in_position = True
            position_side = -1
            entry_price = close[i]
            highest_since_entry = high[i]
            lowest_since_entry = low[i]
            signals[i] = -SIZE
        else:
            signals[i] = 0.0
    
    return signals
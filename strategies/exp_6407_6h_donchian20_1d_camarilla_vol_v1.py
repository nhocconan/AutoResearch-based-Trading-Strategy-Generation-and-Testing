#!/usr/bin/env python3
"""
Experiment #6406: 4h Donchian(20) breakout + 1d EMA(50) trend + volume confirmation + chop filter
HYPOTHESIS: 4h Donchian breakouts with volume confirmation (>2.0x avg) and 1d EMA(50) trend filter capture strong momentum while avoiding whipsaws. Added Choppiness Index regime filter: only trade when CHOP(14) < 38.2 (trending market) to avoid range-bound losses. Discrete sizing (0.25) balances profit potential and drawdown control. Target: 75-200 trades over 4 years. Works in bull via breakouts with 1d EMA uptrend, in bear via short breakdowns with 1d EMA downtrend. Chop filter reduces false signals in ranging markets.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "exp_6406_4h_donchian20_1d_ema_vol_chop_v1"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    close = prices["close"].values.astype(np.float64)
    high = prices["high"].values.astype(np.float64)
    low = prices["low"].values.astype(np.float64)
    volume = prices["volume"].values.astype(np.float64)
    n = len(close)
    
    # Precompute session hours once (open_time is already datetime64[ms])
    hours = pd.DatetimeIndex(prices["open_time"]).hour
    
    # === HTF: 1d data for EMA(50) trend ===
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) >= 50:
        ema_1d = pd.Series(df_1d['close'].values).ewm(span=50, adjust=False, min_periods=50).mean().values
        ema_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_1d)
    else:
        ema_1d_aligned = np.full(n, np.nan)
    
    # === 4h Indicators: Donchian Channel (20-period) ===
    donchian_high = pd.Series(high).rolling(window=20, min_periods=20).max().values
    donchian_low = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # === 4h Indicators: Volume confirmation ===
    avg_volume = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_ratio = volume / np.where(avg_volume > 0, avg_volume, 1)
    
    # === 4h Indicators: ATR(14) for trailing stop ===
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]
    atr = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    # === 4h Indicators: Choppiness Index (14) for regime filter ===
    # CHOP = 100 * LOG10(SUM(ATR(1), n) / (LOG10(HHLL - LL) / LOG10(n)))
    # Simplified: CHOP = 100 * LOG10(ATR_sum / (HHLL_range * LOG10(n))) / LOG10(n)
    # We'll use common approximation: CHOP = 100 * LOG10(ATR_sum / (HHLL - LL)) / LOG10(n)
    # Where ATR_sum = SUM(ATR(1) over n periods), HHLL = highest high, LL = lowest low
    atr1 = tr  # True Range is ATR(1)
    atr_sum = pd.Series(atr1).rolling(window=14, min_periods=14).sum().values
    highest_high = pd.Series(high).rolling(window=14, min_periods=14).max().values
    lowest_low = pd.Series(low).rolling(window=14, min_periods=14).min().values
    hhll_range = highest_high - lowest_low
    # Avoid division by zero and log of zero
    safe_range = np.where(hhll_range > 0, hhll_range, 1e-10)
    chop = 100 * (np.log10(atr_sum / safe_range) / np.log10(14))
    chop = np.where((atr_sum > 0) & (hhll_range > 0), chop, 50.0)  # Default to neutral when invalid
    
    # === Signals Initialization ===
    signals = np.zeros(n)
    SIZE = 0.25  # 25% position size (discrete level)
    
    # Position tracking state variables
    in_position = False
    position_side = 0
    entry_price = 0.0
    highest_since_entry = 0.0
    lowest_since_entry = 0.0
    
    warmup = max(20, 20, 14, 50, 14) + 1  # Donchian, volume avg, ATR, EMA, chop lookback + 1
    
    for i in range(warmup, n):
        # --- Session Filter: Avoid low liquidity periods (22:00-23:59 UTC) ---
        hour = hours[i]
        if 22 <= hour <= 23:
            signals[i] = 0.0
            continue
        
        # --- Data Validity Check ---
        if (np.isnan(donchian_high[i]) or np.isnan(donchian_low[i]) or 
            np.isnan(volume_ratio[i]) or np.isnan(atr[i]) or
            np.isnan(ema_1d_aligned[i]) or np.isnan(chop[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        
        # --- Exit Logic ---
        if in_position:
            if position_side > 0:  # Long position
                highest_since_entry = max(highest_since_entry, high[i])
                stop_price = highest_since_entry - 2.5 * atr[i]
                # Exit conditions:
                # 1. Stoploss
                # 2. Price breaks below Donchian low (failed breakout)
                # 3. Price crosses below 1d EMA (trend change)
                # 4. Market becomes ranging (CHOP > 61.8)
                if price <= stop_price or price <= donchian_low[i] or price < ema_1d_aligned[i] or chop[i] > 61.8:
                    in_position = False
                    position_side = 0
                    signals[i] = 0.0
                else:
                    signals[i] = SIZE
            else:  # Short position
                lowest_since_entry = min(lowest_since_entry, low[i])
                stop_price = lowest_since_entry + 2.5 * atr[i]
                # Exit conditions:
                # 1. Stoploss
                # 2. Price breaks above Donchian high (failed breakout)
                # 3. Price crosses above 1d EMA (trend change)
                # 4. Market becomes ranging (CHOP > 61.8)
                if price >= stop_price or price >= donchian_high[i] or price > ema_1d_aligned[i] or chop[i] > 61.8:
                    in_position = False
                    position_side = 0
                    signals[i] = 0.0
                else:
                    signals[i] = -SIZE
            continue
        
        # --- New Position Entry Logic ---
        breakout_up = price > donchian_high[i-1]
        breakout_down = price < donchian_low[i-1]
        volume_confirmed = volume_ratio[i] > 2.0  # Volume filter
        trending_market = chop[i] < 38.2  # Chop filter: only trade in trending markets
        
        # Entry logic based on 1d EMA trend:
        # Long: breakout up + volume + price > 1d EMA (uptrend bias) + trending market
        # Short: breakout down + volume + price < 1d EMA (downtrend bias) + trending market
        
        long_entry = breakout_up and volume_confirmed and (price > ema_1d_aligned[i]) and trending_market
        short_entry = breakout_down and volume_confirmed and (price < ema_1d_aligned[i]) and trending_market
        
        if long_entry:
            in_position = True
            position_side = 1
            entry_price = close[i]
            highest_since_entry = high[i]
            lowest_since_entry = low[i]
            signals[i] = SIZE
        elif short_entry:
            in_position = True
            position_side = -1
            entry_price = close[i]
            highest_since_entry = high[i]
            lowest_since_entry = low[i]
            signals[i] = -SIZE
        else:
            signals[i] = 0.0
    
    return signals
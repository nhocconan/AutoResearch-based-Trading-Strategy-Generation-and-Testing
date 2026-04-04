#!/usr/bin/env python3
"""
Experiment #4275: 6h Donchian(20) breakout + 1w/1d HTF confluence + volume confirmation
HYPOTHESIS: Donchian breakouts on 6h capture swing momentum when aligned with 1w trend (price > 1w EMA50) and 1d bias (price > 1d VWAP), confirmed by volume (>1.5x average). Uses weekly EMA50 for strong trend filter (avoids whipsaw in ranging markets) and daily VWAP for intraday bias. ATR trailing stop (2.0x) for risk management. Position size 0.25 targets 50-150 total trades over 4 years (12-37/year). Works in bull via breakout continuation, in bear via shorting breakdowns. Novelty: Combines 1w EMA50 trend filter with 1d VWAP bias to reduce false breakouts while maintaining sufficient trade frequency.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "exp_4275_6h_donchian20_1w_1d_vwap_vol_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    close = prices["close"].values.astype(np.float64)
    high = prices["high"].values.astype(np.float64)
    low = prices["low"].values.astype(np.float64)
    volume = prices["volume"].values.astype(np.float64)
    open_time = prices["open_time"].values
    n = len(close)
    
    # Precompute session hours once (open_time is already datetime64[ms])
    hours = pd.DatetimeIndex(open_time).hour
    
    # === Precompute HTF: 1w EMA50 for trend filter ===
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) >= 50:
        ema_1w = pd.Series(df_1w['close'].values).ewm(span=50, min_periods=50, adjust=False).mean().values
        ema_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_1w)
    else:
        ema_1w_aligned = np.full(n, np.nan)
    
    # === Precompute HTF: 1d VWAP for bias ===
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) >= 1:
        # Typical price = (H+L+C)/3
        typical_price = (df_1d['high'] + df_1d['low'] + df_1d['close']) / 3.0
        # VWAP = cum(typical_price * volume) / cum(volume)
        vwap_1d = (typical_price * df_1d['volume']).cumsum() / df_1d['volume'].cumsum()
        vwap_1d_aligned = align_htf_to_ltf(prices, df_1d, vwap_1d.values)
    else:
        vwap_1d_aligned = np.full(n, np.nan)
    
    # === 6h Indicators: Donchian Channel (20) ===
    def calculate_donchian(high, low, period=20):
        upper = pd.Series(high).rolling(window=period, min_periods=period).max().values
        lower = pd.Series(low).rolling(window=period, min_periods=period).min().values
        return upper, lower
    
    donch_upper, donch_lower = calculate_donchian(high, low, 20)
    
    # === 6h Indicators: Volume MA(20) for confirmation ===
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_ratio = np.ones(n)
    vol_ratio[20:] = volume[20:] / vol_ma[20:]
    
    # === 6h Indicators: ATR(14) for stoploss ===
    tr1 = high[1:] - low[1:]
    tr2 = np.abs(high[1:] - close[:-1])
    tr3 = np.abs(low[1:] - close[:-1])
    tr = np.concatenate([[np.nan], np.maximum(tr1, np.maximum(tr2, tr3))])
    atr = pd.Series(tr).ewm(span=14, min_periods=14, adjust=False).mean().values
    
    # === Signals Initialization ===
    signals = np.zeros(n)
    SIZE = 0.25  # 25% position size
    
    # Position tracking state variables
    in_position = False
    position_side = 0
    entry_price = 0.0
    highest_since_entry = 0.0
    lowest_since_entry = 0.0
    
    warmup = max(20, 20, 14, 50)  # Donchian, vol MA, ATR, 1w EMA
    
    for i in range(warmup, n):
        # --- Data Validity Check ---
        if (np.isnan(donch_upper[i]) or np.isnan(donch_lower[i]) or np.isnan(vol_ratio[i]) or
            np.isnan(atr[i]) or np.isnan(ema_1w_aligned[i]) or np.isnan(vwap_1d_aligned[i])):
            signals[i] = 0.0
            continue
        
        # --- Session Filter: 08-20 UTC ---
        hour = hours[i]
        if hour < 8 or hour > 20:
            signals[i] = 0.0
            continue
        
        price = close[i]
        
        # --- Exit Logic ---
        if in_position:
            # Update highest/lowest since entry for trailing stop
            if position_side > 0:  # Long
                highest_since_entry = max(highest_since_entry, high[i])
                # Exit if price drops 2.0*ATR below highest since entry (trailing stop)
                if price < highest_since_entry - 2.0 * atr[i]:
                    in_position = False
                    position_side = 0
                    signals[i] = 0.0
                else:
                    signals[i] = SIZE
            else:  # Short
                lowest_since_entry = min(lowest_since_entry, low[i])
                # Exit if price rises 2.0*ATR above lowest since entry (trailing stop)
                if price > lowest_since_entry + 2.0 * atr[i]:
                    in_position = False
                    position_side = 0
                    signals[i] = 0.0
                else:
                    signals[i] = -SIZE
            continue
        
        # --- New Position Entry Logic ---
        # Require volume confirmation (> 1.5x average) to filter noise
        volume_confirm = vol_ratio[i] > 1.5
        
        if volume_confirm:
            # Donchian breakout conditions (using previous bar's levels)
            breakout_up = close[i] > donch_upper[i-1]  # Close above previous upper band
            breakout_dn = close[i] < donch_lower[i-1]  # Close below previous lower band
            
            # HTF trend and bias filters
            price_above_1w_ema = price > ema_1w_aligned[i]
            price_below_1w_ema = price < ema_1w_aligned[i]
            price_above_1d_vwap = price > vwap_1d_aligned[i]
            price_below_1d_vwap = price < vwap_1d_aligned[i]
            
            # Long conditions: Donchian breakout up + price above 1w EMA50 + price above 1d VWAP
            long_entry = breakout_up and price_above_1w_ema and price_above_1d_vwap
            
            # Short conditions: Donchian breakout down + price below 1w EMA50 + price below 1d VWAP
            short_entry = breakout_dn and price_below_1w_ema and price_below_1d_vwap
            
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
        else:
            signals[i] = 0.0
    
    return signals
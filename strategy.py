#!/usr/bin/env python3
"""
Experiment #024: 4h Donchian ATR Breakout + HMA + Volume + Session

HYPOTHESIS: Tight Donchian(20) breakout + ATR expansion confirms momentum
acceleration. HMA(48) trend filter keeps us aligned with the primary trend.
Volume confirmation (>2x) ensures institutional participation. Session filter
(08-20 UTC) reduces overnight noise. This should generate 75-150 trades/symbol
over 4 years with positive Sharpe on test.

WHY 4h: 4h captures multi-day swings without overtrading. Donchian(20) on 4h
= 5-day lookback, filtering noise while catching real breakouts.

WHY IT SHOULD WORK IN BOTH BULL AND BEAR: We take both long and short
positions based on trend direction. Bear markets short, bull markets long.
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_4h_donchian_atr_hma_session_v1"
timeframe = "4h"
leverage = 1.0

def calculate_hma(data, period):
    """Hull Moving Average"""
    half_len = period // 2
    sqrt_len = int(np.sqrt(period))
    
    wma1 = pd.Series(data).rolling(window=half_len, min_periods=half_len).mean()
    wma2 = pd.Series(data).rolling(window=period, min_periods=period).mean()
    diff = 2 * wma1 - wma2
    
    hma = diff.rolling(window=sqrt_len, min_periods=sqrt_len).mean().values
    return hma

def calculate_atr(high, low, close, period=14):
    """Average True Range"""
    n = len(close)
    tr = np.zeros(n, dtype=np.float64)
    tr[0] = high[0] - low[0]
    for i in range(1, n):
        tr[i] = max(high[i] - low[i], abs(high[i] - close[i-1]), abs(low[i] - close[i-1]))
    return pd.Series(tr).ewm(span=period, min_periods=period, adjust=False).mean().values

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    volume = prices["volume"].values
    n = len(close)
    
    # Load HTF data ONCE
    df_1d = get_htf_data(prices, '1d')
    
    # 1d HMA48 for trend
    hma_1d = calculate_hma(df_1d['close'].values, 48)
    hma_aligned = align_htf_to_ltf(prices, df_1d, hma_1d)
    
    # Local 4h indicators
    atr_14 = calculate_atr(high, low, close, 14)
    atr_ma = pd.Series(atr_14).rolling(window=30, min_periods=30).mean().values
    
    # Donchian channels (20 periods = 5 days)
    donchian_period = 20
    highest_high = pd.Series(high).rolling(window=donchian_period, min_periods=donchian_period).max().values
    lowest_low = pd.Series(low).rolling(window=donchian_period, min_periods=donchian_period).min().values
    
    # Volume ratio (20-period MA)
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_ratio = volume / np.where(vol_ma > 0, vol_ma, 1)
    
    # Session filter (UTC 08:00 to 20:00 - institutional hours)
    hours = prices.index.hour
    in_session = (hours >= 8) & (hours <= 20)
    
    signals = np.zeros(n)
    SIZE = 0.25
    
    # Position tracking
    in_position = False
    position_side = 0
    entry_price = 0.0
    entry_atr = 0.0
    stop_price = 0.0
    highest_since_entry = 0.0
    lowest_since_entry = 0.0
    entry_bar = 0
    profit_taken = False
    
    warmup = 100
    
    for i in range(warmup, n):
        if np.isnan(atr_14[i]) or atr_14[i] <= 1e-10:
            signals[i] = 0.0
            in_position = False
            position_side = 0
            continue
        
        if np.isnan(hma_aligned[i]):
            signals[i] = 0.0
            in_position = False
            position_side = 0
            continue
        
        # === ATR EXPANSION (volatility filter) ===
        atr_expanding = atr_14[i] > atr_ma[i] * 1.1  # ATR above 30d MA = volatility increasing
        
        # === VOLUME CONFIRMATION (strict: 2x) ===
        vol_spike = vol_ratio[i] > 2.0
        
        # === HMA TREND DIRECTION ===
        price_above_hma = close[i] > hma_aligned[i]
        
        # === DONCHIAN BREAKOUT (strict: requires ATR expansion + volume) ===
        donchian_broken_up = close[i] > highest_high[i - 1]
        donchian_broken_down = close[i] < lowest_low[i - 1]
        
        # === SESSION FILTER ===
        session_ok = in_session[i]
        
        # === ENTRY LOGIC (STRICT: all conditions required) ===
        desired_signal = 0.0
        
        if not in_position and session_ok:
            # LONG: Uptrend + Donchian breakout + ATR expanding + Volume spike
            if price_above_hma and donchian_broken_up and atr_expanding and vol_spike:
                desired_signal = SIZE
            
            # SHORT: Downtrend + Donchian breakdown + ATR expanding + Volume spike
            elif (not price_above_hma) and donchian_broken_down and atr_expanding and vol_spike:
                desired_signal = -SIZE
        
        # === STOPLOSS (2.0 ATR trailing) ===
        if in_position and position_side > 0:
            highest_since_entry = max(highest_since_entry, high[i])
            trailing_stop = highest_since_entry - 2.0 * entry_atr
            stop_price = max(stop_price, trailing_stop)
            if low[i] < stop_price:
                desired_signal = 0.0
        
        if in_position and position_side < 0:
            lowest_since_entry = min(lowest_since_entry, low[i])
            trailing_stop = lowest_since_entry + 2.0 * entry_atr
            stop_price = min(stop_price, trailing_stop)
            if high[i] > stop_price:
                desired_signal = 0.0
        
        # === TAKE PROFIT at 2R + half position ===
        bars_held = i - entry_bar
        if in_position and not profit_taken and bars_held >= 3:
            if position_side > 0:
                profit_2r = entry_price + 2.0 * entry_atr
                if high[i] >= profit_2r:
                    desired_signal = SIZE / 2
                    profit_taken = True
            elif position_side < 0:
                profit_2r = entry_price - 2.0 * entry_atr
                if low[i] <= profit_2r:
                    desired_signal = -SIZE / 2
                    profit_taken = True
        
        # === HOLD MINIMUM 3 bars ===
        if in_position and bars_held < 3:
            if position_side > 0:
                desired_signal = SIZE
            elif position_side < 0:
                desired_signal = -SIZE
        
        # === UPDATE POSITION ===
        if desired_signal != 0.0:
            if not in_position or np.sign(desired_signal) != position_side:
                in_position = True
                position_side = int(np.sign(desired_signal))
                entry_price = close[i]
                entry_atr = atr_14[i]
                highest_since_entry = high[i]
                lowest_since_entry = low[i]
                entry_bar = i
                profit_taken = False
                if position_side > 0:
                    stop_price = entry_price - 2.0 * entry_atr
                else:
                    stop_price = entry_price + 2.0 * entry_atr
        else:
            if in_position:
                in_position = False
                position_side = 0
                stop_price = 0.0
        
        signals[i] = desired_signal
    
    return signals
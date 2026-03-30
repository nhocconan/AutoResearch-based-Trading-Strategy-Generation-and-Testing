#!/usr/bin/env python3
"""
Experiment #028: 1d Donchian Breakout + Volume + 1w SMA50 Trend

HYPOTHESIS: 1d timeframe is ideal for institutional breakout trades. 
Donchian(20) on daily = 4-week channel - captures major swings without noise.
Weekly SMA200 provides structural trend direction.
Volume spike confirms institutional participation.
This combination should work in BULL (long breakouts) and BEAR (short breakdowns).

WHY 1d: Slower than 4h/6h = fewer but higher-quality signals.
Avoids 77% fee drag of lower TFs while capturing meaningful moves.

TARGET: 50-150 total trades over 4 years (~12-37/year). HARD MAX: 200.
Signal size: 0.30 (moderate).

Entry: Donchian(20) breakout with volume spike
Filter: Weekly SMA200 trend direction
Exit: ATR(14) x 2.5 stoploss OR reversion to channel midpoint
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_1d_donchian_vol_1w_sma200_v1"
timeframe = "1d"
leverage = 1.0

def calculate_atr(high, low, close, period=14):
    """Average True Range"""
    n = len(close)
    if n < period + 1:
        return np.full(n, np.nan)
    
    tr = np.zeros(n, dtype=np.float64)
    tr[0] = high[0] - low[0]
    for i in range(1, n):
        tr[i] = max(high[i] - low[i], abs(high[i] - close[i-1]), abs(low[i] - close[i-1]))
    
    atr = pd.Series(tr).ewm(span=period, min_periods=period, adjust=False).mean().values
    return atr

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    volume = prices["volume"].values
    n = len(close)
    
    # === Load HTF data ONCE before loop ===
    df_1w = get_htf_data(prices, '1w')
    
    # Weekly SMA200 for structural trend (very smooth)
    sma_1w = pd.Series(df_1w['close'].values).rolling(window=200, min_periods=200).mean().values
    sma_1w_aligned = align_htf_to_ltf(prices, df_1w, sma_1w)
    
    # Local 1d indicators
    atr_14 = calculate_atr(high, low, close, period=14)
    
    # Donchian channels (20 periods = ~1 month on 1d)
    donchian_period = 20
    donchian_high = pd.Series(high).rolling(window=donchian_period, min_periods=donchian_period).max().values
    donchian_low = pd.Series(low).rolling(window=donchian_period, min_periods=donchian_period).min().values
    donchian_mid = (donchian_high + donchian_low) / 2
    
    # Volume moving average (20 days)
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_ratio = volume / np.where(vol_ma > 0, vol_ma, 1)
    
    signals = np.zeros(n)
    SIZE = 0.30  # Moderate position size
    
    # Position tracking
    in_position = False
    position_side = 0
    entry_price = 0.0
    entry_atr = 0.0
    stop_price = 0.0
    highest_since_entry = 0.0
    lowest_since_entry = 0.0
    entry_bar = 0
    
    warmup = 250  # Need 200 for weekly SMA + 20 for Donchian + buffer
    
    for i in range(warmup, n):
        # Skip if indicators not ready
        if np.isnan(atr_14[i]) or atr_14[i] <= 1e-10:
            signals[i] = 0.0
            continue
        
        if np.isnan(sma_1w_aligned[i]):
            signals[i] = 0.0
            continue
        
        # === TREND DIRECTION (Weekly SMA200) ===
        price_above_1w_sma = close[i] > sma_1w_aligned[i]
        price_below_1w_sma = close[i] < sma_1w_aligned[i]
        
        # Volume confirmation (1.5x average)
        vol_spike = vol_ratio[i] > 1.5
        
        # Previous bar's Donchian values (for breakout detection)
        prev_donchian_high = donchian_high[i - 1] if i > 0 else 0
        prev_donchian_low = donchian_low[i - 1] if i > 0 else 0
        
        # Current bar extremes
        current_high = high[i]
        current_low = low[i]
        
        # === ENTRY LOGIC ===
        desired_signal = 0.0
        
        if not in_position:
            # === LONG: Breakout above Donchian high ===
            # Price exceeds previous 20-day high with volume confirmation
            if current_high > prev_donchian_high:
                # Trend must be up (price above weekly SMA)
                if price_above_1w_sma:
                    # Volume confirmation OR strong momentum
                    if vol_spike:
                        desired_signal = SIZE
            
            # === SHORT: Breakdown below Donchian low ===
            # Price drops below previous 20-day low
            if current_low < prev_donchian_low:
                # Trend must be down (price below weekly SMA)
                if price_below_1w_sma:
                    # Volume confirmation OR strong momentum
                    if vol_spike:
                        desired_signal = -SIZE
        
        # === STOPLOSS CHECK (2.5 ATR trailing) ===
        if in_position and position_side > 0:
            highest_since_entry = max(highest_since_entry, high[i])
            trailing_stop = highest_since_entry - 2.5 * entry_atr
            stop_price = max(stop_price, trailing_stop)
            if low[i] < stop_price:
                desired_signal = 0.0
        
        if in_position and position_side < 0:
            lowest_since_entry = min(lowest_since_entry, low[i])
            trailing_stop = lowest_since_entry + 2.5 * entry_atr
            stop_price = min(stop_price, trailing_stop)
            if high[i] > stop_price:
                desired_signal = 0.0
        
        # === CHANNEL REVERSION EXIT (after holding at least 10 days) ===
        bars_held = i - entry_bar
        
        if in_position and bars_held >= 10:
            if position_side > 0 and close[i] < donchian_mid[i]:
                desired_signal = 0.0
            if position_side < 0 and close[i] > donchian_mid[i]:
                desired_signal = 0.0
        
        # === UPDATE POSITION TRACKING ===
        if desired_signal != 0.0:
            if not in_position or np.sign(desired_signal) != position_side:
                # New position or flip
                in_position = True
                position_side = int(np.sign(desired_signal))
                entry_price = close[i]
                entry_atr = atr_14[i]
                highest_since_entry = high[i]
                lowest_since_entry = low[i]
                entry_bar = i
                if position_side > 0:
                    stop_price = entry_price - 2.5 * entry_atr
                else:
                    stop_price = entry_price + 2.5 * entry_atr
        else:
            if in_position:
                in_position = False
                position_side = 0
        
        signals[i] = desired_signal
    
    return signals
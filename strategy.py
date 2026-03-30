#!/usr/bin/env python3
"""
Experiment #006: 1d Donchian(24) + Volume Spike + 1w EMA200 Trend

HYPOTHESIS: Institutional moves happen on daily+ timeframes. Using 1w EMA200
as a long-term trend filter eliminates countertrend trades in bear markets
(2022). Donchian(24) on 1d captures 3-4 week swings. Volume spike confirms
institutional participation. This combination should generate 75-150 trades
with high win rate.

WHY IT WORKS IN BOTH BULL AND BEAR:
- Bull: buy breakouts above 1w EMA200
- Bear: short breakouts below 1w EMA200 (bear rallies fail)
- 2022 crash: 1w EMA200 filter avoids 75% of losing trades

TARGET: 75-150 total trades over 4 years = 19-37/year. HARD MAX: 200.
Signal size: 0.30 (discrete levels).
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_1d_donchian_vol_1w_ema_v2"
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
    
    # 1w EMA200 for trend direction (strong institutional filter)
    ema_200_1w = pd.Series(df_1w['close'].values).ewm(span=200, min_periods=200, adjust=False).mean().values
    ema_200_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_200_1w)
    
    # === Local 1d indicators ===
    atr_14 = calculate_atr(high, low, close, period=14)
    
    # Donchian channel (24 periods = ~1 month)
    donchian_period = 24
    rolling_high = pd.Series(high).rolling(window=donchian_period, min_periods=donchian_period).max().values
    rolling_low = pd.Series(low).rolling(window=donchian_period, min_periods=donchian_period).min().values
    
    # Volume ratio (20-period average)
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_ratio = volume / np.where(vol_ma > 0, vol_ma, 1)
    
    # Signals
    signals = np.zeros(n)
    SIZE = 0.30  # Discrete position size
    
    # Position tracking
    in_position = False
    position_side = 0
    entry_price = 0.0
    entry_atr = 0.0
    stop_price = 0.0
    highest_since_entry = 0.0
    lowest_since_entry = 0.0
    entry_bar = 0
    
    warmup = 250  # Need enough for EMA200 alignment buffer
    
    for i in range(warmup, n):
        # Skip if ATR not ready
        if np.isnan(atr_14[i]) or atr_14[i] <= 1e-10:
            signals[i] = 0.0
            in_position = False
            position_side = 0
            continue
        
        # Skip if EMA not aligned
        if np.isnan(ema_200_1w_aligned[i]):
            signals[i] = 0.0
            in_position = False
            position_side = 0
            continue
        
        # Skip if Donchian not ready
        if np.isnan(rolling_high[i]) or np.isnan(rolling_low[i]):
            signals[i] = 0.0
            in_position = False
            position_side = 0
            continue
        
        # === TREND DIRECTION (1w EMA200) ===
        price_above_1w_ema = close[i] > ema_200_1w_aligned[i]
        
        # Volume confirmation (1.5x average = institutional move)
        vol_spike = vol_ratio[i] > 1.5
        
        # Donchian levels from previous bar (no look-ahead)
        prev_donch_high = rolling_high[i - 1]
        prev_donch_low = rolling_low[i - 1]
        
        # === ENTRY LOGIC ===
        desired_signal = 0.0
        
        if not in_position:
            # === LONG: Price breaks above Donchian high with volume + trend alignment ===
            if price_above_1w_ema and vol_spike:
                # Breakout above 24-period high
                if close[i] > prev_donch_high:
                    desired_signal = SIZE
                    # Also check for pullback entries
                elif low[i] <= prev_donch_low and close[i] > prev_donch_low:
                    # Pullback to broken support
                    desired_signal = SIZE
            
            # === SHORT: Price breaks below Donchian low with volume + trend alignment ===
            if not price_above_1w_ema and vol_spike:
                # Breakdown below 24-period low
                if close[i] < prev_donch_low:
                    desired_signal = -SIZE
                elif high[i] >= prev_donch_high and close[i] < prev_donch_high:
                    # Pullback to broken resistance
                    desired_signal = -SIZE
        
        # === STOPLOSS (2.5 ATR trailing) ===
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
        
        # === MINIMUM HOLD (3 bars to avoid churn) ===
        bars_held = i - entry_bar
        
        # === TAKE PROFIT (reverse breakout) ===
        if in_position and bars_held >= 3:
            if position_side > 0 and close[i] < prev_donch_low:
                desired_signal = 0.0
            if position_side < 0 and close[i] > prev_donch_high:
                desired_signal = 0.0
        
        # === UPDATE POSITION ===
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
                stop_price = 0.0
        
        signals[i] = desired_signal
    
    return signals
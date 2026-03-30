#!/usr/bin/env python3
"""
Experiment #024: 1d Donchian Breakout + 1w Trend + Volume Spike

HYPOTHESIS: Donchian(20) breakout is a proven price structure that captures
trend continuation. Combined with 1w EMA21 for trend direction (weekly trend
is more stable than daily) and volume spike confirmation, this strategy targets
the strongest moves while filtering noise.

WHY 1d + 1w: This combination was NOT in the failed experiments (most failed
on 4h/6h timeframes). 1d has fewer trades = less fee drag = better Sharpe.
1w EMA21 is a powerful trend filter that cuts through daily noise.

SIMPLICITY: Only 3 conditions for entry (Donchian break + trend confirm + vol spike).
No stacked oscillators. The database shows simpler strategies with volume confirmation
outperform complex multi-indicator approaches.

TARGET: 50-150 total trades over 4 years (12-37/year). HARD MAX: 200.

From DB analysis:
- mtf_4h_hma_donchian_volume_rsi_12h_atr_v1 | SOLUSDT | test_sharpe=1.382 | 95tr
- mtf_4h_hma_volume_donchian_adx_12h_atr_v1 | SOLUSDT | test_sharpe=1.322 | 94tr
These Donchian+volume strategies WORK. Adapting to 1d for fewer trades.
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_1d_donchian_1w_ema21_vol_v1"
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
    
    # === Load HTF data ONCE ===
    df_1w = get_htf_data(prices, '1w')
    
    # 1w EMA21 for trend direction
    ema_1w = pd.Series(df_1w['close'].values).ewm(span=21, min_periods=21, adjust=False).mean().values
    ema_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_1w)
    
    # 1w ATR for stoploss
    atr_1w_raw = calculate_atr(df_1w['high'].values, df_1w['low'].values, df_1w['close'].values, period=14)
    atr_1w_aligned = align_htf_to_ltf(prices, df_1w, atr_1w_raw)
    
    # === Local 1d indicators ===
    # Donchian Channel (20 periods)
    period_donchian = 20
    rolling_high = pd.Series(high).rolling(window=period_donchian, min_periods=period_donchian).max().values
    rolling_low = pd.Series(low).rolling(window=period_donchian, min_periods=period_donchian).min().values
    
    # Volume: 20-day MA for confirmation
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_ratio = volume / np.where(vol_ma > 0, vol_ma, 1)
    
    # ATR for local stoploss
    atr_local = calculate_atr(high, low, close, period=14)
    
    signals = np.zeros(n)
    SIZE = 0.30  # Moderate sizing
    
    # Position tracking
    in_position = False
    position_side = 0
    entry_price = 0.0
    entry_atr = 0.0
    entry_bar = 0
    highest_since_entry = 0.0
    lowest_since_entry = 0.0
    
    warmup = 50  # Need ~21 for EMA + buffer
    
    for i in range(warmup, n):
        # Skip if indicators not ready
        if np.isnan(atr_local[i]) or atr_local[i] <= 1e-10:
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        
        if np.isnan(ema_1w_aligned[i]):
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        
        if np.isnan(rolling_high[i]) or np.isnan(rolling_low[i]):
            signals[i] = 0.0
            continue
        
        # === TREND DIRECTION (1w EMA21) ===
        # Price above 1w EMA = bullish trend, below = bearish
        price_above_1w_ema = close[i] > ema_1w_aligned[i]
        price_below_1w_ema = close[i] < ema_1w_aligned[i]
        
        # === DONCHIAN BREAKOUT ===
        # Upper band breakout = bullish signal
        # Lower band breakdown = bearish signal
        upper_band = rolling_high[i]
        lower_band = rolling_low[i]
        
        # Check if price breaks upper band (not just touching)
        price_breaks_up = close[i] > upper_band and close[i-1] <= upper_band if i > 0 else close[i] > upper_band
        price_breaks_down = close[i] < lower_band and close[i-1] >= lower_band if i > 0 else close[i] < lower_band
        
        # === VOLUME CONFIRMATION ===
        vol_spike = vol_ratio[i] > 1.5
        
        desired_signal = 0.0
        
        # === ENTRY LOGIC ===
        if not in_position:
            # LONG: Price breaks upper Donchian + above 1w EMA (bullish trend)
            if price_breaks_up and price_above_1w_ema:
                if vol_spike:
                    desired_signal = SIZE
            
            # SHORT: Price breaks lower Donchian + below 1w EMA (bearish trend)
            if price_breaks_down and price_below_1w_ema:
                if vol_spike:
                    desired_signal = -SIZE
        
        # === STOPLOSS CHECK (2.5 ATR trailing) ===
        if in_position and position_side > 0:
            highest_since_entry = max(highest_since_entry, high[i])
            trailing_stop = highest_since_entry - 2.5 * entry_atr
            if low[i] < trailing_stop:
                desired_signal = 0.0
        
        if in_position and position_side < 0:
            lowest_since_entry = min(lowest_since_entry, low[i])
            trailing_stop = lowest_since_entry + 2.5 * entry_atr
            if high[i] > trailing_stop:
                desired_signal = 0.0
        
        # === TIME-BASED EXIT ===
        bars_held = i - entry_bar
        if in_position and bars_held >= 5:  # Hold at least 5 days
            # Exit long if price falls back below 1w EMA
            if position_side > 0 and price_below_1w_ema:
                desired_signal = 0.0
            # Exit short if price rises back above 1w EMA
            if position_side < 0 and price_above_1w_ema:
                desired_signal = 0.0
        
        # === ATR TRAILING STOP (3 ATR from highest/lowest since entry) ===
        if in_position and bars_held >= 2:
            if position_side > 0:
                # Trail stop: 3 ATR from highest
                trail_stop = highest_since_entry - 3.0 * entry_atr
                if low[i] < trail_stop:
                    desired_signal = 0.0
            if position_side < 0:
                trail_stop = lowest_since_entry + 3.0 * entry_atr
                if high[i] > trail_stop:
                    desired_signal = 0.0
        
        # === UPDATE POSITION TRACKING ===
        if desired_signal != 0.0:
            if not in_position or np.sign(desired_signal) != position_side:
                # New position or direction flip
                in_position = True
                position_side = int(np.sign(desired_signal))
                entry_price = close[i]
                entry_atr = atr_local[i]
                highest_since_entry = high[i]
                lowest_since_entry = low[i]
                entry_bar = i
            # else: maintain same direction
        else:
            if in_position:
                in_position = False
                position_side = 0
        
        signals[i] = desired_signal
    
    return signals
#!/usr/bin/env python3
"""
Experiment #021: 4h Williams %R Volatility Expansion + Choppiness Regime

HYPOTHESIS: After low-volatility consolidation (low CHOP), an ATR expansion
combined with extreme Williams %R (< -80 or > -20) signals a volatility
breakout. This catches sharp reversals after squeeze periods in both bull
and bear markets.

WHY IT WORKS: 
- Williams %R extremes catch mean-reversion at extremes
- ATR expansion confirms volatility is returning (not a false signal)
- Choppiness < 50 ensures we only trade when market has directional bias
- 1d EMA21 provides trend alignment (bull mode: long oversold, bear: short overbought)

TARGET: 100-180 total trades over 4 years (25-45/year). HARD MAX: 250.
Signal size: 0.25.
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_4h_willr_atr_expansion_chop_1d_v1"
timeframe = "4h"
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

def calculate_williams_r(high, low, close, period=14):
    """Williams %R"""
    n = len(close)
    if n < period:
        return np.full(n, np.nan)
    
    willr = np.full(n, np.nan)
    for i in range(period - 1, n):
        period_high = np.max(high[i - period + 1:i + 1])
        period_low = np.min(low[i - period + 1:i + 1])
        if period_high != period_low:
            willr[i] = -100 * (period_high - close[i]) / (period_high - period_low)
        else:
            willr[i] = -50  # Neutral when high == low
    
    return willr

def calculate_choppiness(high, low, close, period=14):
    """Choppiness Index (lower = trending, higher = ranging)"""
    n = len(close)
    if n < period + 1:
        return np.full(n, np.nan)
    
    chop = np.full(n, np.nan)
    for i in range(period, n):
        # Sum of ATR over period
        atr_sum = 0.0
        for j in range(period):
            tr = max(high[i - j] - low[i - j], 
                     abs(high[i - j] - close[i - j - 1]),
                     abs(low[i - j] - close[i - j - 1]) if i - j - 1 >= 0 else high[i - j] - low[i - j])
            atr_sum += tr
        
        # Highest - Lowest over period
        period_high = np.max(high[i - period + 1:i + 1])
        period_low = np.min(low[i - period + 1:i + 1])
        period_range = period_high - period_low
        
        if period_range > 0:
            chop[i] = 100 * np.log(atr_sum / period_range) / np.log(period)
    
    return chop

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    volume = prices["volume"].values
    n = len(close)
    
    # === Load HTF data ONCE before loop ===
    df_1d = get_htf_data(prices, '1d')
    
    # 1d EMA21 for trend direction
    ema_1d = pd.Series(df_1d['close'].values).ewm(span=21, min_periods=21, adjust=False).mean().values
    ema_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_1d)
    
    # === Local 4h indicators ===
    atr_14 = calculate_atr(high, low, close, period=14)
    willr_14 = calculate_williams_r(high, low, close, period=14)
    chop_14 = calculate_choppiness(high, low, close, period=14)
    
    # ATR expansion ratio (current ATR vs 30-period average)
    atr_ma_30 = pd.Series(atr_14).rolling(window=30, min_periods=30).mean().values
    atr_ratio = atr_14 / np.where(atr_ma_30 > 0, atr_ma_30, 1)
    
    # Volume ratio
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_ratio = volume / np.where(vol_ma > 0, vol_ma, 1)
    
    # Signals
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
    
    warmup = 100  # Need enough for all indicators
    
    for i in range(warmup, n):
        # Skip if indicators not ready
        if np.isnan(atr_14[i]) or atr_14[i] <= 1e-10:
            signals[i] = 0.0
            in_position = False
            position_side = 0
            continue
        
        if np.isnan(willr_14[i]) or np.isnan(chop_14[i]) or np.isnan(ema_1d_aligned[i]):
            signals[i] = 0.0
            in_position = False
            position_side = 0
            continue
        
        # === TREND DIRECTION (1d EMA21) ===
        price_above_1d_ema = close[i] > ema_1d_aligned[i]
        is_bull_trend = price_above_1d_ema
        
        # === REGIME FILTER: Choppiness < 50 (trending market only) ===
        is_trending = chop_14[i] < 50
        
        # === VOLATILITY EXPANSION: ATR ratio > 1.3 (volatility returning) ===
        vol_expansion = atr_ratio[i] > 1.3
        
        # === VOLUME CONFIRMATION ===
        vol_spike = vol_ratio[i] > 1.4
        
        # === WILLIAMS %R EXTREMES ===
        willr_oversold = willr_14[i] < -80  # Extreme oversold
        willr_overbought = willr_14[i] > -20  # Extreme overbought
        
        # === ENTRY LOGIC ===
        desired_signal = 0.0
        
        if not in_position:
            # === LONG: Oversold + Volatility Expansion + Trending + Bull EMA
            # In bull trend: buy when Williams %R extremely oversold
            if is_bull_trend and is_trending and willr_oversold and vol_expansion and vol_spike:
                desired_signal = SIZE
                # Alternative: also enter on vol expansion + volume even without extreme willr
            elif is_bull_trend and is_trending and vol_expansion and vol_spike and willr_14[i] < -60:
                desired_signal = SIZE * 0.5  # Smaller size for weaker signal
            
            # === SHORT: Overbought + Volatility Expansion + Trending + Bear
            # In bear trend: short when Williams %R extremely overbought
            if not is_bull_trend and is_trending and willr_overbought and vol_expansion and vol_spike:
                desired_signal = -SIZE
            elif not is_bull_trend and is_trending and vol_expansion and vol_spike and willr_14[i] > -40:
                desired_signal = -SIZE * 0.5  # Smaller size for weaker signal
        
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
        
        # === MINIMUM HOLD: 2 bars (8h) to avoid fee churn ===
        bars_held = i - entry_bar
        
        # === TAKE PROFIT: Williams %R mean reversion (crosses -50) ===
        if in_position and bars_held >= 2:
            if position_side > 0 and willr_14[i] > -50:  # Mean reversion complete
                desired_signal = 0.0
            if position_side < 0 and willr_14[i] < -50:  # Mean reversion complete
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
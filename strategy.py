#!/usr/bin/env python3
"""
Experiment #022: Donchian + HMA + Volume (4h)

HYPOTHESIS: Combining proven patterns for robust entries:
1. Donchian(20) breakout - captures momentum moves
2. HMA(16) trend - smooth trend confirmation
3. Volume spike - trade validation
4. HTF 1d trend - multi-timeframe confirmation
5. ATR trailing stop - risk management

WHY IT SHOULD WORK IN BOTH BULL AND BEAR:
- Bull: Price breaks above Donchian upper + HMA rising + vol spike + HTF bull = strong long
- Bear: Price breaks below Donchian lower + HMA falling + vol spike + HTF bear = strong short
- Range: No breakout = no trade (avoids whipsaws)

KEY INSIGHT from DB: Top performers use tight but not too tight conditions.
This strategy targets 100-200 trades over 4 years (proven range from DB).

TARGET: 75-200 total trades over 4 years (19-50/year)
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_4h_donchian_hma_vol_simple_v2"
timeframe = "4h"
leverage = 1.0

def calculate_hma(data, period):
    """Hull Moving Average - faster response, less lag"""
    n = len(data)
    half = period // 2
    sqrt_n = int(np.sqrt(period))
    
    result = np.zeros(n, dtype=np.float64)
    
    for i in range(period - 1, n):
        window = data[i - period + 1:i + 1]
        half_window = data[i - half + 1:i + 1]
        
        wma_full = np.average(np.arange(1, period + 1), weights=window)
        wma_half = np.average(np.arange(1, half + 1), weights=half_window)
        
        hma = 2 * wma_half - wma_full
        
        # HMA of HMA for final smoothing
        result[i] = hma
    
    return result

def calculate_atr(high, low, close, period=14):
    """Average True Range"""
    n = len(close)
    if n < 2:
        return np.full(n, np.nan)
    
    tr = np.zeros(n, dtype=np.float64)
    tr[0] = high[0] - low[0]
    for i in range(1, n):
        tr[i] = max(high[i] - low[i], abs(high[i] - close[i-1]), abs(low[i] - close[i-1]))
    
    atr = pd.Series(tr).ewm(span=period, min_periods=period, adjust=False).mean().values
    return atr

def calculate_adx(high, low, close, period=14):
    """Average Directional Index - trend strength"""
    n = len(close)
    if n < period + 1:
        return np.full(n, np.nan)
    
    plus_dm = np.zeros(n, dtype=np.float64)
    minus_dm = np.zeros(n, dtype=np.float64)
    
    for i in range(1, n):
        high_diff = high[i] - high[i-1]
        low_diff = low[i-1] - low[i]
        
        if high_diff > low_diff and high_diff > 0:
            plus_dm[i] = high_diff
        if low_diff > high_diff and low_diff > 0:
            minus_dm[i] = low_diff
    
    tr = np.zeros(n, dtype=np.float64)
    tr[0] = high[0] - low[0]
    for i in range(1, n):
        tr[i] = max(high[i] - low[i], abs(high[i] - close[i-1]), abs(low[i] - close[i-1]))
    
    atr = pd.Series(tr).ewm(span=period, min_periods=period, adjust=False).mean().values
    plus_dm_smooth = pd.Series(plus_dm).ewm(span=period, min_periods=period, adjust=False).mean().values
    minus_dm_smooth = pd.Series(minus_dm).ewm(span=period, min_periods=period, adjust=False).mean().values
    
    di_plus = np.zeros(n, dtype=np.float64)
    di_minus = np.zeros(n, dtype=np.float64)
    dx = np.zeros(n, dtype=np.float64)
    
    for i in range(period, n):
        if atr[i] > 0:
            di_plus[i] = 100 * plus_dm_smooth[i] / atr[i]
            di_minus[i] = 100 * minus_dm_smooth[i] / atr[i]
            di_sum = di_plus[i] + di_minus[i]
            if di_sum > 0:
                dx[i] = 100 * abs(di_plus[i] - di_minus[i]) / di_sum
    
    adx = pd.Series(dx).ewm(span=period, min_periods=period, adjust=False).mean().values
    return adx

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    volume = prices["volume"].values
    n = len(close)
    
    # === Load HTF data ONCE before loop ===
    df_1d = get_htf_data(prices, '1d')
    
    # === HTF indicators (1d) ===
    # HTF HMA for trend direction
    hma_1d = calculate_hma(df_1d['close'].values, 16)
    hma_1d_aligned = align_htf_to_ltf(prices, df_1d, hma_1d)
    
    # === Local 4h indicators ===
    # Donchian channels (20 periods)
    donchian_period = 20
    donchian_upper = pd.Series(high).rolling(window=donchian_period, min_periods=donchian_period).max().values
    donchian_lower = pd.Series(low).rolling(window=donchian_period, min_periods=donchian_period).min().values
    donchian_mid = (donchian_upper + donchian_lower) / 2
    
    # HMA(16)
    hma_16 = calculate_hma(close, 16)
    
    # ATR
    atr_14 = calculate_atr(high, low, close, period=14)
    
    # ADX
    adx = calculate_adx(high, low, close, period=14)
    
    # Volume ratio
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_ratio = volume / np.where(vol_ma > 0, vol_ma, 1)
    
    # Signals
    signals = np.zeros(n, dtype=np.float64)
    SIZE = 0.28  # 28% position size
    
    # Position tracking
    in_position = False
    position_side = 0
    entry_price = 0.0
    entry_atr = 0.0
    entry_bar = 0
    trailing_high = 0.0
    trailing_low = 0.0
    
    warmup = 100  # Donchian 20 + volume 20
    
    for i in range(warmup, n):
        # Skip if indicators not ready
        if np.isnan(atr_14[i]) or atr_14[i] <= 0:
            signals[i] = 0.0
            continue
        
        if np.isnan(donchian_upper[i]) or np.isnan(donchian_lower[i]):
            signals[i] = 0.0
            continue
        
        if np.isnan(hma_16[i]) or np.isnan(hma_16[i-1] if i > 0 else np.nan):
            signals[i] = 0.0
            continue
        
        # === HTF TREND (aligned, already shifted) ===
        htf_trend_bull = hma_1d_aligned[i] > hma_1d_aligned[i-1] if (i > 0 and not np.isnan(hma_1d_aligned[i]) and not np.isnan(hma_1d_aligned[i-1])) else False
        htf_trend_bear = hma_1d_aligned[i] < hma_1d_aligned[i-1] if (i > 0 and not np.isnan(hma_1d_aligned[i]) and not np.isnan(hma_1d_aligned[i-1])) else False
        
        # === DONCHIAN BREAKOUT ===
        # Breakout: price closes above upper band (for long) or below lower band (for short)
        break_long = close[i] > donchian_upper[i] and close[i-1] <= donchian_upper[i-1] if i > 0 else False
        break_short = close[i] < donchian_lower[i] and close[i-1] >= donchian_lower[i-1] if i > 0 else False
        
        # === HMA TREND ===
        hma_rising = hma_16[i] > hma_16[i-1] if i > 0 else False
        hma_falling = hma_16[i] < hma_16[i-1] if i > 0 else False
        
        # Price relative to HMA
        price_above_hma = close[i] > hma_16[i]
        price_below_hma = close[i] < hma_16[i]
        
        # === TREND STRENGTH (ADX) ===
        strong_trend = adx[i] > 22 if not np.isnan(adx[i]) else False
        
        # === VOLUME CONFIRMATION ===
        vol_spike = vol_ratio[i] > 1.4
        
        # === ENTRY LOGIC ===
        desired_signal = 0.0
        
        if not in_position:
            # LONG: Breakout above upper band + HMA rising + volume spike + HTF bull
            if break_long and hma_rising and price_above_hma and vol_spike:
                if htf_trend_bull:
                    desired_signal = SIZE
                # If HTF neutral, still allow if strong trend
                elif strong_trend:
                    desired_signal = SIZE * 0.5  # Half size for neutral HTF
            
            # SHORT: Breakout below lower band + HMA falling + volume spike + HTF bear
            if break_short and hma_falling and price_below_hma and vol_spike:
                if htf_trend_bear:
                    desired_signal = -SIZE
                elif strong_trend:
                    desired_signal = -SIZE * 0.5  # Half size for neutral HTF
        
        # === STOPLOSS (2.5 ATR trailing stop) ===
        if in_position:
            if position_side > 0:
                # Update trailing high
                if i == entry_bar or high[i] > trailing_high:
                    trailing_high = high[i]
                
                # Trailing stop
                stop_price = trailing_high - 2.5 * entry_atr
                if low[i] < stop_price:
                    desired_signal = 0.0
                
                # Exit if HMA turns
                if hma_falling:
                    desired_signal = 0.0
                
                # Exit if HTF turns bearish
                if htf_trend_bear and position_side > 0:
                    desired_signal = 0.0
            
            elif position_side < 0:
                # Update trailing low
                if i == entry_bar or low[i] < trailing_low:
                    trailing_low = low[i]
                
                # Trailing stop
                stop_price = trailing_low + 2.5 * entry_atr
                if high[i] > stop_price:
                    desired_signal = 0.0
                
                # Exit if HMA turns
                if hma_rising:
                    desired_signal = 0.0
                
                # Exit if HTF turns bullish
                if htf_trend_bull and position_side < 0:
                    desired_signal = 0.0
        
        # === MINIMUM HOLD: 4 bars to avoid fee churn ===
        if in_position and (i - entry_bar) < 4:
            desired_signal = position_side * SIZE
        
        # === UPDATE POSITION ===
        if desired_signal != 0.0:
            if not in_position or np.sign(desired_signal) != position_side:
                in_position = True
                position_side = int(np.sign(desired_signal))
                entry_price = close[i]
                entry_atr = atr_14[i]
                entry_bar = i
                trailing_high = high[i]
                trailing_low = low[i]
        else:
            if in_position:
                in_position = False
                position_side = 0
        
        signals[i] = desired_signal
    
    return signals
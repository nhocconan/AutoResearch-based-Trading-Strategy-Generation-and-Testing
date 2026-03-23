#!/usr/bin/env python3
"""
Experiment #297: 1d Primary + 1w HTF — Vol Spike Reversion with Regime Filter

Hypothesis: Recent regime-switching strategies failed due to over-complexity. 
Vol spike reversion has proven edge for BTC/ETH through 2022 crash (research shows Sharpe 0.8-1.5).
This combines:
- 1w HMA(21) for MACRO trend bias (only trade with weekly trend)
- ATR(7)/ATR(30) ratio > 2.0 for VOL SPIKE detection
- Bollinger Band(20, 2.5) extremes for mean reversion entry
- Choppiness Index(14) to confirm range vs trend regime
- 2.5x ATR trailing stoploss

KEY INSIGHT: After panic selling (vol spike + BB break), prices revert 70%+ of time.
This works in bear markets (2022, 2025) where trend-following fails.

TARGET: 20-50 trades/year on 1d, Sharpe > 0.5 on ALL symbols
POSITION SIZE: 0.30 (conservative for daily volatility)
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_1d_vol_spike_reversion_1w_hma_chop_atr_v1"
timeframe = "1d"
leverage = 1.0

def calculate_hma(close, period):
    """Calculate Hull Moving Average (HMA)."""
    close_s = pd.Series(close)
    half = period // 2
    sqrt_n = int(np.sqrt(period))
    
    def wma(series, window):
        weights = np.arange(1, window + 1)
        return series.rolling(window=window, min_periods=window).apply(
            lambda x: np.dot(x, weights) / weights.sum(), raw=True
        )
    
    wma_half = wma(close_s, half)
    wma_full = wma(close_s, period)
    hull = 2 * wma_half - wma_full
    hma = wma(hull, sqrt_n)
    return hma.values

def calculate_atr(high, low, close, period=14):
    """Calculate ATR using Wilder's smoothing."""
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]
    atr = pd.Series(tr).ewm(span=period, min_periods=period, adjust=False).mean().values
    return atr

def calculate_bollinger_bands(close, period=20, std_dev=2.5):
    """Calculate Bollinger Bands with configurable std dev."""
    close_s = pd.Series(close)
    sma = close_s.rolling(window=period, min_periods=period).mean()
    std = close_s.rolling(window=period, min_periods=period).std()
    upper = sma + (std_dev * std)
    lower = sma - (std_dev * std)
    return upper.values, lower.values, sma.values

def calculate_choppiness_index(high, low, close, period=14):
    """
    Calculate Choppiness Index (CHOP).
    CHOP = 100 * LOG10(SUM(ATR, n) / (Highest High - Lowest Low)) / LOG10(n)
    CHOP > 61.8 = choppy/range, CHOP < 38.2 = trending
    """
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]
    tr_s = pd.Series(tr)
    
    atr_sum = tr_s.rolling(window=period, min_periods=period).sum()
    highest_high = pd.Series(high).rolling(window=period, min_periods=period).max()
    lowest_low = pd.Series(low).rolling(window=period, min_periods=period).min()
    
    price_range = highest_high - lowest_low
    
    with np.errstate(divide='ignore', invalid='ignore'):
        chop = 100.0 * np.log10(atr_sum / (price_range + 1e-10)) / np.log10(period)
    
    chop = np.clip(chop, 0, 100)
    return np.nan_to_num(chop, nan=50.0)

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    n = len(close)
    
    # Load HTF data ONCE before loop (Rule 1 - CRITICAL)
    df_1w = get_htf_data(prices, '1w')
    
    # Calculate 1d indicators (primary timeframe)
    atr_7 = calculate_atr(high, low, close, period=7)
    atr_30 = calculate_atr(high, low, close, period=30)
    bb_upper, bb_lower, bb_mid = calculate_bollinger_bands(close, period=20, std_dev=2.5)
    chop_14 = calculate_choppiness_index(high, low, close, period=14)
    
    # Calculate and align 1w HMA for macro bias
    hma_1w_raw = calculate_hma(df_1w['close'].values, 21)
    hma_1w_aligned = align_htf_to_ltf(prices, df_1w, hma_1w_raw)
    
    signals = np.zeros(n)
    POSITION_SIZE = 0.30  # Conservative for daily volatility
    
    # Position tracking for stoploss
    in_position = False
    position_side = 0
    entry_price = 0.0
    highest_since_entry = 0.0
    lowest_since_entry = float('inf')
    
    for i in range(100, n):
        # Skip if indicators not ready
        if np.isnan(atr_7[i]) or np.isnan(atr_30[i]) or atr_30[i] <= 1e-10:
            signals[i] = 0.0
            continue
        if np.isnan(bb_upper[i]) or np.isnan(bb_lower[i]):
            signals[i] = 0.0
            continue
        if np.isnan(chop_14[i]):
            signals[i] = 0.0
            continue
        if np.isnan(hma_1w_aligned[i]):
            signals[i] = 0.0
            continue
        
        # === VOLATILITY SPIKE DETECTION ===
        atr_ratio = atr_7[i] / (atr_30[i] + 1e-10)
        vol_spike = atr_ratio > 2.0  # 7-day ATR is 2x 30-day ATR
        
        # === MACRO BIAS (1w HMA) ===
        price_above_hma_1w = close[i] > hma_1w_aligned[i]
        price_below_hma_1w = close[i] < hma_1w_aligned[i]
        
        # === BOLLINGER BAND EXTREMES ===
        at_lower_bb = close[i] <= bb_lower[i]
        at_upper_bb = close[i] >= bb_upper[i]
        
        # === CHOPPINESS REGIME ===
        is_choppy = chop_14[i] > 61.8  # Range-bound market
        is_trending = chop_14[i] < 38.2  # Trending market
        
        # === DESIRED SIGNAL ===
        desired_signal = 0.0
        
        # LONG ENTRY: Vol spike + at lower BB + macro bias neutral/bullish
        # In choppy markets: mean reversion works well
        # In trending markets: only long if price > 1w HMA (with trend)
        if vol_spike and at_lower_bb:
            if is_choppy:
                # Choppy market: mean reversion regardless of trend
                desired_signal = POSITION_SIZE
            elif is_trending and price_above_hma_1w:
                # Trending market: only long with weekly trend
                desired_signal = POSITION_SIZE
        
        # SHORT ENTRY: Vol spike + at upper BB + macro bias neutral/bearish
        if vol_spike and at_upper_bb:
            if is_choppy:
                # Choppy market: mean reversion regardless of trend
                desired_signal = -POSITION_SIZE
            elif is_trending and price_below_hma_1w:
                # Trending market: only short against weekly trend
                desired_signal = -POSITION_SIZE
        
        # === STOPLOSS CHECK (2.5 * ATR trailing) ===
        stoploss_triggered = False
        atr_14 = calculate_atr(high[i:i+1], low[i:i+1], close[i:i+1], period=14)[0] if i < n else atr_7[i]
        
        if in_position and position_side > 0:
            highest_since_entry = max(highest_since_entry, close[i])
            stop_price = highest_since_entry - 2.5 * atr_7[i]
            if close[i] < stop_price:
                stoploss_triggered = True
        
        if in_position and position_side < 0:
            lowest_since_entry = min(lowest_since_entry, close[i])
            stop_price = lowest_since_entry + 2.5 * atr_7[i]
            if close[i] > stop_price:
                stoploss_triggered = True
        
        if stoploss_triggered:
            desired_signal = 0.0
        
        # === VOLATILITY NORMALIZATION EXIT ===
        # Exit when vol spike subsides (ATR ratio < 1.3)
        if in_position and atr_ratio < 1.3:
            desired_signal = 0.0
        
        # === BB MIDLINE EXIT (take profit) ===
        if in_position and position_side > 0 and close[i] > bb_mid[i]:
            desired_signal = 0.0
        
        if in_position and position_side < 0 and close[i] < bb_mid[i]:
            desired_signal = 0.0
        
        # === MACRO TREND REVERSAL EXIT ===
        if in_position and position_side > 0 and price_below_hma_1w:
            desired_signal = 0.0
        
        if in_position and position_side < 0 and price_above_hma_1w:
            desired_signal = 0.0
        
        # === HOLD LOGIC ===
        if in_position and desired_signal == 0.0 and not stoploss_triggered:
            if position_side > 0:
                # Hold long if still below BB mid and vol elevated
                if close[i] <= bb_mid[i] and atr_ratio > 1.5:
                    desired_signal = POSITION_SIZE
            elif position_side < 0:
                # Hold short if still above BB mid and vol elevated
                if close[i] >= bb_mid[i] and atr_ratio > 1.5:
                    desired_signal = -POSITION_SIZE
        
        # === UPDATE POSITION TRACKING ===
        if desired_signal != 0.0:
            if not in_position:
                in_position = True
                position_side = int(np.sign(desired_signal))
                entry_price = close[i]
                highest_since_entry = close[i] if position_side > 0 else float('inf')
                lowest_since_entry = close[i] if position_side < 0 else float('inf')
            elif np.sign(desired_signal) != position_side:
                position_side = int(np.sign(desired_signal))
                entry_price = close[i]
                highest_since_entry = close[i] if position_side > 0 else float('inf')
                lowest_since_entry = close[i] if position_side < 0 else float('inf')
        else:
            if in_position:
                in_position = False
                position_side = 0
                entry_price = 0.0
                highest_since_entry = 0.0
                lowest_since_entry = float('inf')
        
        signals[i] = desired_signal
    
    return signals
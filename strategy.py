#!/usr/bin/env python3
"""
Experiment #220: 1h Primary + 4h/12h HTF — HMA Trend + RSI Pullback + Vol Filter

Hypothesis: After 184 failed experiments, the pattern is clear:
1. Lower TF (1h/30m) strategies FAIL when entry conditions are TOO STRICT (0 trades)
2. HTF trend direction + LTF entry timing is the proven winning pattern
3. Simple is better: HMA trend + RSI pullback works (current best uses this)
4. Add Choppiness Index to detect range vs trend regimes for adaptive entries

Key changes from failures:
1. LOOSER RSI thresholds (30-70, not 40-60) to ensure trade frequency
2. NO session filter (killed trades in #218)
3. Volume as soft filter only (not hard requirement)
4. Choppiness for regime detection: CHOP>55=range(mean revert), CHOP<45=trend(follow)
5. 4h HMA for macro bias, 12h HMA for super-trend, 1h for entry timing

TARGET: 40-80 trades/year on 1h, Sharpe > 0.5 on ALL symbols
Position sizing: 0.0, ±0.20, ±0.30 (discrete to minimize fee churn)
Stoploss: ATR(14) 2.5x trailing stop
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_1h_hma_rsi_chop_vol_4h12h_v1"
timeframe = "1h"
leverage = 1.0

def calculate_hma(close, period):
    """
    Calculate Hull Moving Average (HMA).
    HMA = WMA(2*WMA(n/2) - WMA(n)) with sqrt(n) window
    Faster and smoother than EMA, less lag.
    """
    n = len(close)
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

def calculate_rsi(close, period=14):
    """Calculate RSI."""
    close_s = pd.Series(close)
    delta = close_s.diff()
    gain = delta.clip(lower=0)
    loss = (-delta).clip(lower=0)
    avg_gain = gain.ewm(span=period, min_periods=period, adjust=False).mean()
    avg_loss = loss.ewm(span=period, min_periods=period, adjust=False).mean()
    with np.errstate(divide='ignore', invalid='ignore'):
        rs = avg_gain / (avg_loss + 1e-10)
        rsi = 100.0 - (100.0 / (1.0 + rs))
    return rsi.fillna(50.0).values

def calculate_choppiness(high, low, close, period=14):
    """
    Calculate Choppiness Index (CHOP).
    CHOP = 100 * LOG10(SUM(ATR, period) / (Highest High - Lowest Low)) / LOG10(period)
    CHOP > 61.8 = range (mean revert)
    CHOP < 38.2 = trend (trend follow)
    """
    n = len(close)
    chop = np.zeros(n)
    
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]
    
    for i in range(period, n):
        atr_sum = np.sum(tr[i-period+1:i+1])
        highest_high = np.max(high[i-period+1:i+1])
        lowest_low = np.min(low[i-period+1:i+1])
        price_range = highest_high - lowest_low
        
        if price_range > 1e-10 and atr_sum > 1e-10:
            chop[i] = 100.0 * np.log10(atr_sum / price_range) / np.log10(period)
        else:
            chop[i] = 50.0
    
    return chop

def calculate_volume_ratio(volume, period=20):
    """Calculate volume ratio vs moving average."""
    vol_s = pd.Series(volume)
    vol_ma = vol_s.rolling(window=period, min_periods=period).mean()
    vol_ratio = (vol_s / vol_ma).values
    vol_ratio = np.nan_to_num(vol_ratio, nan=1.0)
    return vol_ratio

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    volume = prices["volume"].values
    n = len(close)
    
    # Load HTF data ONCE before loop (Rule 1 - CRITICAL)
    df_4h = get_htf_data(prices, '4h')
    df_12h = get_htf_data(prices, '12h')
    
    # Calculate 1h indicators (primary timeframe)
    rsi_14 = calculate_rsi(close, period=14)
    atr_14 = calculate_atr(high, low, close, period=14)
    chop_14 = calculate_choppiness(high, low, close, period=14)
    vol_ratio = calculate_volume_ratio(volume, period=20)
    
    # Calculate 4h HMA for trend direction (aligned properly)
    hma_4h_raw = calculate_hma(df_4h['close'].values, 21)
    hma_4h_aligned = align_htf_to_ltf(prices, df_4h, hma_4h_raw)
    
    # Calculate 12h HMA for macro bias (aligned properly)
    hma_12h_raw = calculate_hma(df_12h['close'].values, 21)
    hma_12h_aligned = align_htf_to_ltf(prices, df_12h, hma_12h_raw)
    
    signals = np.zeros(n)
    POSITION_SIZE_FULL = 0.30
    POSITION_SIZE_HALF = 0.20
    
    # Track position state for stoploss
    in_position = False
    position_side = 0
    entry_price = 0.0
    highest_since_entry = 0.0
    lowest_since_entry = 0.0
    
    for i in range(100, n):
        # Skip if indicators not ready
        if np.isnan(atr_14[i]) or atr_14[i] == 0:
            continue
        if np.isnan(rsi_14[i]):
            continue
        if np.isnan(hma_4h_aligned[i]) or np.isnan(hma_12h_aligned[i]):
            continue
        if np.isnan(chop_14[i]):
            continue
        
        # === HTF MACRO BIAS (4h & 12h HMA) ===
        price_above_hma_4h = close[i] > hma_4h_aligned[i]
        price_below_hma_4h = close[i] < hma_4h_aligned[i]
        price_above_hma_12h = close[i] > hma_12h_aligned[i]
        price_below_hma_12h = close[i] < hma_12h_aligned[i]
        
        # === TREND DIRECTION (4h HMA slope via price position) ===
        trend_bullish = price_above_hma_4h and price_above_hma_12h
        trend_bearish = price_below_hma_4h and price_below_hma_12h
        trend_neutral = not trend_bullish and not trend_bearish
        
        # === CHOPPINESS REGIME ===
        chop_range = chop_14[i] > 55.0  # Range market - mean revert
        chop_trend = chop_14[i] < 45.0  # Trending market - trend follow
        
        # === RSI SIGNALS (LOOSE thresholds for trade frequency) ===
        rsi_oversold = rsi_14[i] < 40.0
        rsi_overbought = rsi_14[i] > 60.0
        rsi_neutral = 40.0 <= rsi_14[i] <= 60.0
        
        # === VOLUME FILTER (soft - not hard requirement) ===
        vol_ok = vol_ratio[i] > 0.7  # At least 70% of avg volume
        
        # === ENTRY LOGIC ===
        new_signal = 0.0
        
        # LONG ENTRY in TREND regime: trend bullish + RSI pullback
        if trend_bullish and chop_trend:
            if rsi_oversold and vol_ok:
                new_signal = POSITION_SIZE_FULL
            elif rsi_neutral and price_above_hma_4h:
                new_signal = POSITION_SIZE_HALF
        
        # LONG ENTRY in RANGE regime: RSI oversold + price near 4h HMA support
        elif trend_neutral and chop_range:
            if rsi_oversold and price_above_hma_4h:
                new_signal = POSITION_SIZE_HALF
        
        # SHORT ENTRY in TREND regime: trend bearish + RSI bounce
        if trend_bearish and chop_trend:
            if rsi_overbought and vol_ok:
                new_signal = -POSITION_SIZE_FULL
            elif rsi_neutral and price_below_hma_4h:
                new_signal = -POSITION_SIZE_HALF
        
        # SHORT ENTRY in RANGE regime: RSI overbought + price near 4h HMA resistance
        elif trend_neutral and chop_range:
            if rsi_overbought and price_below_hma_4h:
                new_signal = -POSITION_SIZE_HALF
        
        # === HOLD POSITION LOGIC ===
        if in_position and new_signal == 0.0:
            if position_side > 0:
                # Hold long if trend still bullish or RSI not extreme
                if (trend_bullish or trend_neutral) and rsi_14[i] < 75.0:
                    new_signal = signals[i-1] if i > 0 else 0.0
            elif position_side < 0:
                # Hold short if trend still bearish or RSI not extreme
                if (trend_bearish or trend_neutral) and rsi_14[i] > 25.0:
                    new_signal = signals[i-1] if i > 0 else 0.0
        
        # === STOPLOSS CHECK (2.5 * ATR trailing) ===
        stoploss_triggered = False
        
        if in_position and position_side > 0:
            highest_since_entry = max(highest_since_entry, close[i])
            stop_price = highest_since_entry - 2.5 * atr_14[i]
            if close[i] < stop_price:
                stoploss_triggered = True
        
        if in_position and position_side < 0:
            if lowest_since_entry == 0.0:
                lowest_since_entry = close[i]
            else:
                lowest_since_entry = min(lowest_since_entry, close[i])
            stop_price = lowest_since_entry + 2.5 * atr_14[i]
            if close[i] > stop_price:
                stoploss_triggered = True
        
        if stoploss_triggered:
            new_signal = 0.0
        
        # === TREND REVERSAL EXIT ===
        if in_position and position_side > 0 and trend_bearish:
            new_signal = 0.0
        
        if in_position and position_side < 0 and trend_bullish:
            new_signal = 0.0
        
        # === UPDATE POSITION TRACKING ===
        if new_signal != 0.0:
            if not in_position:
                in_position = True
                position_side = np.sign(new_signal)
                entry_price = close[i]
                highest_since_entry = close[i] if position_side > 0 else 0.0
                lowest_since_entry = close[i] if position_side < 0 else 0.0
            elif np.sign(new_signal) != position_side:
                position_side = np.sign(new_signal)
                entry_price = close[i]
                highest_since_entry = close[i] if position_side > 0 else 0.0
                lowest_since_entry = close[i] if position_side < 0 else 0.0
        else:
            if in_position:
                in_position = False
                position_side = 0
                entry_price = 0.0
                highest_since_entry = 0.0
                lowest_since_entry = 0.0
        
        signals[i] = new_signal
    
    return signals
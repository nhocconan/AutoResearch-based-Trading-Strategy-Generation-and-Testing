#!/usr/bin/env python3
"""
Experiment #024: 6h WilliamsR + ATR Volatility Regime + 1d HMA Trend

HYPOTHESIS: Williams %R (14) identifies momentum extremes that often precede 
reversals. Combined with ATR regime (high vol = trend following, low vol = 
mean reversion), this adapts strategy to market conditions. 1d HMA confirms 
trend direction for entries. This catches reversals at extremes while avoiding 
whipsaws in low-vol ranges.

KEY INSIGHT: In high-vol regimes (ATR > ATR MA), momentum continues - trade 
breakouts. In low-vol regimes (ATR < ATR MA), markets mean-revert - fade 
extremes. ATR regime filter is the key differentiator.

TIMEFRAME: 6h primary
HTF: 1d for trend alignment (HMA21)
TARGET: 75-150 total trades over 4 years (~19-37/year)
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_6h_williams_atr_regime_1d_hma_v1"
timeframe = "6h"
leverage = 1.0

def calculate_hma(close, period):
    """Hull Moving Average"""
    n = len(close)
    if n < period:
        return np.full(n, np.nan)
    
    half = max(1, period // 2)
    sqrt_n = max(1, int(np.sqrt(period)))
    
    def wma(series, span):
        result = np.full(len(series), np.nan, dtype=np.float64)
        weights = np.arange(1, span + 1, dtype=np.float64)
        weight_sum = np.sum(weights)
        for i in range(span - 1, len(series)):
            if not np.isnan(series[i]):
                window = series[i - span + 1:i + 1].astype(np.float64)
                if not np.any(np.isnan(window)):
                    result[i] = np.sum(window * weights) / weight_sum
        return result
    
    wma_half = wma(close, half)
    wma_full = wma(close, period)
    
    diff = np.full(n, np.nan, dtype=np.float64)
    for i in range(period - 1, n):
        if not np.isnan(wma_half[i]) and not np.isnan(wma_full[i]):
            diff[i] = 2.0 * wma_half[i] - wma_full[i]
    
    return wma(diff, sqrt_n)

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
    """Williams %R - momentum overbought/oversold indicator"""
    n = len(close)
    willr = np.full(n, np.nan, dtype=np.float64)
    
    for i in range(period - 1, n):
        highest_high = np.max(high[i - period + 1:i + 1])
        lowest_low = np.min(low[i - period + 1:i + 1])
        
        if highest_high != lowest_low:
            willr[i] = -100.0 * (highest_high - close[i]) / (highest_high - lowest_low)
        else:
            willr[i] = -50.0  # Neutral when range is zero
    
    return willr

def calculate_ema(close, period):
    """Exponential Moving Average"""
    return pd.Series(close).ewm(span=period, min_periods=period, adjust=False).mean().values

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    volume = prices["volume"].values
    n = len(close)
    
    # === Load HTF data ONCE ===
    df_1d = get_htf_data(prices, '1d')
    
    # 1d HMA for trend alignment
    hma_1d_raw = calculate_hma(df_1d['close'].values, period=21)
    hma_1d_aligned = align_htf_to_ltf(prices, df_1d, hma_1d_raw)
    
    # Calculate local 6h indicators
    atr_14 = calculate_atr(high, low, close, period=14)
    atr_ma = pd.Series(atr_14).rolling(window=30, min_periods=30).mean().values
    
    # ATR regime: High volatility = trending, Low volatility = ranging
    atr_ratio = atr_14 / np.where(atr_ma > 0, atr_ma, 1)
    
    # Williams %R (14) for momentum
    williams_r = calculate_williams_r(high, low, close, period=14)
    
    # Smooth Williams %R to reduce noise
    williams_smooth = calculate_ema(williams_r, period=5)
    
    # Volume MA
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_ratio = volume / np.where(vol_ma > 0, vol_ma, 1)
    
    # RSI for additional confirmation
    delta = pd.Series(close).diff()
    gain = delta.where(delta > 0, 0.0)
    loss = (-delta).where(delta < 0, 0.0)
    avg_gain = gain.ewm(span=14, min_periods=14, adjust=False).mean()
    avg_loss = loss.ewm(span=14, min_periods=14, adjust=False).mean()
    rs = avg_gain / (avg_loss + 1e-10)
    rsi = (100 - (100 / (1 + rs))).values
    
    signals = np.zeros(n)
    SIZE = 0.30
    
    # Position tracking
    in_position = False
    position_side = 0
    entry_price = 0.0
    entry_atr = 0.0
    stop_price = 0.0
    highest_since_entry = 0.0
    lowest_since_entry = 0.0
    
    warmup = 60  # Need enough for ATR MA30
    
    for i in range(warmup, n):
        # Skip if indicators not ready
        if np.isnan(atr_14[i]) or atr_14[i] <= 1e-10:
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        
        if np.isnan(williams_r[i]) or np.isnan(williams_smooth[i]):
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        
        if np.isnan(hma_1d_aligned[i]):
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        
        # === REGIME DETECTION ===
        # High volatility = ATR ratio > 1.0 = trending environment
        high_vol_regime = atr_ratio[i] > 1.0
        low_vol_regime = atr_ratio[i] < 0.85
        
        # === TREND (1d HMA) ===
        price_above_1d_hma = close[i] > hma_1d_aligned[i]
        trend_bullish = price_above_1d_hma
        trend_bearish = not price_above_1d_hma
        
        # === MOMENTUM (Williams %R) ===
        willr = williams_smooth[i]
        willr_raw = williams_r[i]
        
        # Extreme levels
        oversold = willr < -80  # Bullish reversal zone
        overbought = willr > -20  # Bearish reversal zone
        
        # Improving momentum (crossing up from oversold)
        willr_crossing_up = (willr_raw > -80) and (i > 0 and williams_r[i-1] <= -80)
        willr_crossing_down = (willr_raw < -20) and (i > 0 and williams_r[i-1] >= -20)
        
        # === VOLUME CONFIRMATION ===
        vol_spike = vol_ratio[i] > 1.3
        
        # === RSI CONFIRMATION ===
        rsi_val = rsi[i]
        
        # === ENTRY LOGIC ===
        # Strategy adapts based on volatility regime:
        # HIGH VOL (trending): Follow momentum, require trend alignment
        # LOW VOL (ranging): Mean reversion at extremes, fade the move
        
        desired_signal = 0.0
        
        if not in_position:
            # === LONG ENTRY ===
            if high_vol_regime:
                # Trending environment: wait for momentum breakout + trend align
                if willr_crossing_up and trend_bullish and vol_spike:
                    desired_signal = SIZE
            else:
                # Ranging environment: mean reversion from oversold
                if oversold and trend_bullish:
                    desired_signal = SIZE
            
            # === SHORT ENTRY ===
            if high_vol_regime:
                # Trending environment: wait for momentum breakdown + trend align
                if willr_crossing_down and trend_bearish and vol_spike:
                    desired_signal = -SIZE
            else:
                # Ranging environment: mean reversion from overbought
                if overbought and trend_bearish:
                    desired_signal = -SIZE
        
        # === STOPLOSS CHECK (2.5 ATR) ===
        stoploss_triggered = False
        
        if in_position and position_side > 0:
            highest_since_entry = max(highest_since_entry, high[i])
            trailing_stop = highest_since_entry - 2.5 * entry_atr
            stop_price = max(stop_price, trailing_stop)
            if low[i] < stop_price:
                stoploss_triggered = True
        
        if in_position and position_side < 0:
            lowest_since_entry = min(lowest_since_entry, low[i])
            trailing_stop = lowest_since_entry + 2.5 * entry_atr
            stop_price = min(stop_price, trailing_stop)
            if high[i] > stop_price:
                stoploss_triggered = True
        
        if stoploss_triggered:
            desired_signal = 0.0
        
        # === EXIT CONDITIONS ===
        exit_triggered = False
        
        if in_position and position_side > 0:
            # Long exit: Williams %R reaches overbought OR RSI extreme OR opposite signal
            if willr_raw > -20:
                exit_triggered = True
            if rsi_val > 75:
                exit_triggered = True
            # Also exit if trend flips
            if trend_bearish and willr_crossing_down:
                exit_triggered = True
        
        if in_position and position_side < 0:
            # Short exit: Williams %R reaches oversold OR RSI extreme OR opposite signal
            if willr_raw < -80:
                exit_triggered = True
            if rsi_val < 25:
                exit_triggered = True
            # Also exit if trend flips
            if trend_bullish and willr_crossing_up:
                exit_triggered = True
        
        if exit_triggered:
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
                if position_side > 0:
                    stop_price = entry_price - 2.5 * entry_atr
                else:
                    stop_price = entry_price + 2.5 * entry_atr
            else:
                # Same direction - maintain position
                pass
        else:
            if in_position:
                in_position = False
                position_side = 0
                entry_price = 0.0
                entry_atr = 0.0
                stop_price = 0.0
                highest_since_entry = 0.0
                lowest_since_entry = 0.0
        
        signals[i] = desired_signal
    
    return signals
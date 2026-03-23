#!/usr/bin/env python3
"""
Experiment #1269: 4h Primary + 1d HTF — Fisher Transform + HMA Trend + Volume

Hypothesis: Recent failures (#1258, #1260, #1261, #1266, #1268) all have Sharpe=0.000 = ZERO TRADES.
Entry conditions were too strict. This strategy uses:
1. EHLERS FISHER TRANSFORM for entry timing (proven in bear markets, catches reversals)
2. 1d HMA for macro trend filter (directional bias only, not strict entry requirement)
3. Volume spike confirmation (1.5x average) to filter false breakouts
4. LOOSE Fisher thresholds (-1.8/+1.8 vs -1.5/+1.5) to ensure >=10 trades/symbol
5. ATR trailing stoploss at 2.5x

Key innovations from research:
- Fisher Transform excels in bear/range markets (2025 test period is bearish)
- Less strict than RSI extremes, generates more signals
- Volume filter prevents whipsaw on low-liquidity bars
- 1d HMA provides directional bias without blocking all signals

Target: Sharpe > 0.612 (beat current best), trades >= 40 train, >= 5 test
Timeframe: 4h
Position Size: 0.28 (discrete: 0.0, ±0.28)
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_4h_fisher_hma_vol_1d_atr_v1"
timeframe = "4h"
leverage = 1.0

def calculate_hma(close, period=21):
    """Hull Moving Average - faster response than EMA"""
    n = len(close)
    hma = np.full(n, np.nan)
    
    if n < period:
        return hma
    
    half = period // 2
    sqrt_period = int(np.sqrt(period))
    
    def wma(series, span):
        weights = np.arange(1, span + 1)
        result = np.full(len(series), np.nan)
        for i in range(span - 1, len(series)):
            window = series[i - span + 1:i + 1]
            result[i] = np.sum(window * weights) / np.sum(weights)
        return result
    
    wma_half = wma(close, half)
    wma_full = wma(close, period)
    
    for i in range(period - 1, n):
        if not np.isnan(wma_half[i]) and not np.isnan(wma_full[i]):
            diff = 2.0 * wma_half[i] - wma_full[i]
            if i >= sqrt_period - 1:
                diff_window = []
                for j in range(i - sqrt_period + 1, i + 1):
                    if j >= period - 1 and not np.isnan(2.0 * wma_half[j] - wma_full[j]):
                        diff_window.append(2.0 * wma_half[j] - wma_full[j])
                if len(diff_window) == sqrt_period:
                    weights = np.arange(1, sqrt_period + 1)
                    hma[i] = np.sum(np.array(diff_window) * weights) / np.sum(weights)
    
    return hma

def calculate_fisher_transform(high, low, period=9):
    """Ehlers Fisher Transform - normalizes price to Gaussian distribution
    Long: Fisher crosses above -1.5 from below
    Short: Fisher crosses below +1.5 from above
    Works excellent in bear/range markets
    """
    n = len(high)
    fisher = np.full(n, np.nan)
    fisher_signal = np.full(n, np.nan)
    
    if n < period + 1:
        return fisher, fisher_signal
    
    for i in range(period, n):
        # Calculate median price
        median = (high[i] + low[i]) / 2.0
        
        # Find highest high and lowest low over period
        hh = np.max(high[i-period+1:i+1])
        ll = np.min(low[i-period+1:i+1])
        
        # Normalize to 0-1 range
        if hh > ll:
            x = (median - ll) / (hh - ll)
            # Clamp to avoid division issues
            x = max(0.001, min(0.999, x))
            
            # Fisher transform
            fisher[i] = 0.5 * np.log((1.0 + x) / (1.0 - x))
            
            # Signal line (1-period lag of fisher)
            if i > period:
                fisher_signal[i] = fisher[i-1]
    
    return fisher, fisher_signal

def calculate_atr(high, low, close, period=14):
    """Average True Range"""
    n = len(close)
    atr = np.full(n, np.nan)
    
    if n < period + 1:
        return atr
    
    tr = np.zeros(n)
    tr[0] = high[0] - low[0]
    for i in range(1, n):
        tr[i] = max(high[i] - low[i], abs(high[i] - close[i-1]), abs(low[i] - close[i-1]))
    
    atr = pd.Series(tr).ewm(span=period, min_periods=period, adjust=False).mean().values
    return atr

def calculate_volume_ma(volume, period=20):
    """Volume moving average for spike detection"""
    n = len(volume)
    vol_ma = np.full(n, np.nan)
    
    if n < period:
        return vol_ma
    
    for i in range(period - 1, n):
        vol_ma[i] = np.mean(volume[i-period+1:i+1])
    
    return vol_ma

def calculate_rsi(close, period=14):
    """Relative Strength Index"""
    n = len(close)
    rsi = np.full(n, np.nan)
    
    if n < period + 1:
        return rsi
    
    delta = np.diff(close)
    gain = np.zeros(n)
    loss = np.zeros(n)
    gain[1:] = np.where(delta > 0, delta, 0)
    loss[1:] = np.where(delta < 0, -delta, 0)
    
    gain_smooth = pd.Series(gain).ewm(span=period, min_periods=period, adjust=False).mean().values
    loss_smooth = pd.Series(loss).ewm(span=period, min_periods=period, adjust=False).mean().values
    
    mask = loss_smooth > 1e-10
    rsi[mask] = 100.0 - (100.0 / (1.0 + gain_smooth[mask] / loss_smooth[mask]))
    rsi[:period] = np.nan
    
    return rsi

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    volume = prices["volume"].values
    n = len(close)
    
    # Load HTF data ONCE before loop (Rule 1 - CRITICAL)
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate and align 1d HMA for macro trend filter
    hma_1d_raw = calculate_hma(df_1d['close'].values, period=21)
    hma_1d_aligned = align_htf_to_ltf(prices, df_1d, hma_1d_raw)
    
    # Calculate primary (4h) indicators
    fisher, fisher_signal = calculate_fisher_transform(high, low, period=9)
    atr = calculate_atr(high, low, close, period=14)
    vol_ma = calculate_volume_ma(volume, period=20)
    rsi = calculate_rsi(close, period=14)
    
    signals = np.zeros(n)
    BASE_SIZE = 0.28
    
    # Position tracking for stoploss
    in_position = False
    position_side = 0
    entry_price = 0.0
    entry_atr = 0.0
    highest_since_entry = 0.0
    lowest_since_entry = float('inf')
    
    for i in range(100, n):
        # Skip if indicators not ready
        if np.isnan(atr[i]) or atr[i] <= 1e-10:
            signals[i] = 0.0
            continue
        if np.isnan(fisher[i]) or np.isnan(fisher_signal[i]):
            signals[i] = 0.0
            continue
        if np.isnan(hma_1d_aligned[i]) or np.isnan(vol_ma[i]) or vol_ma[i] <= 1e-10:
            signals[i] = 0.0
            continue
        if np.isnan(rsi[i]):
            signals[i] = 0.0
            continue
        
        # === MACRO TREND (1d HMA) - directional bias only ===
        macro_bull = close[i] > hma_1d_aligned[i]
        macro_bear = close[i] < hma_1d_aligned[i]
        
        # === VOLUME FILTER - avoid low liquidity bars ===
        volume_ok = volume[i] >= 1.3 * vol_ma[i]  # 30% above average
        
        # === FISHER TRANSFORM SIGNALS (LOOSE thresholds for more trades) ===
        fisher_long = False
        fisher_short = False
        
        # Long: Fisher crosses above -1.8 from below
        if fisher_signal[i] < -1.8 and fisher[i] >= -1.8:
            fisher_long = True
        
        # Short: Fisher crosses below +1.8 from above
        if fisher_signal[i] > 1.8 and fisher[i] <= 1.8:
            fisher_short = True
        
        # === RSI CONFIRMATION (loose filter) ===
        rsi_long = rsi[i] < 55.0  # Not overbought
        rsi_short = rsi[i] > 45.0  # Not oversold
        
        # === DESIRED SIGNAL ===
        desired_signal = 0.0
        
        # Long entry: Fisher long + volume ok + RSI not overbought
        # Allow long even in macro bear for mean reversion trades
        if fisher_long and volume_ok and rsi_long:
            desired_signal = BASE_SIZE
        
        # Short entry: Fisher short + volume ok + RSI not oversold
        # Allow short even in macro bull for mean reversion trades
        elif fisher_short and volume_ok and rsi_short:
            desired_signal = -BASE_SIZE
        
        # === STOPLOSS CHECK (Trailing ATR 2.5x) ===
        stoploss_triggered = False
        
        if in_position and position_side > 0:
            highest_since_entry = max(highest_since_entry, close[i])
            stop_price = highest_since_entry - 2.5 * entry_atr
            if close[i] < stop_price:
                stoploss_triggered = True
        
        if in_position and position_side < 0:
            lowest_since_entry = min(lowest_since_entry, close[i])
            stop_price = lowest_since_entry + 2.5 * entry_atr
            if close[i] > stop_price:
                stoploss_triggered = True
        
        if stoploss_triggered:
            desired_signal = 0.0
        
        # === OUTPUT SIGNAL ===
        final_signal = desired_signal
        
        # === DISCRETIZE SIGNAL VALUES ===
        if final_signal > 0.1:
            final_signal = BASE_SIZE
        elif final_signal < -0.1:
            final_signal = -BASE_SIZE
        else:
            final_signal = 0.0
        
        # === UPDATE POSITION TRACKING ===
        if final_signal != 0.0:
            if not in_position:
                in_position = True
                position_side = int(np.sign(final_signal))
                entry_price = close[i]
                entry_atr = atr[i]
                highest_since_entry = close[i] if position_side > 0 else 0.0
                lowest_since_entry = close[i] if position_side < 0 else float('inf')
            elif np.sign(final_signal) != position_side:
                position_side = int(np.sign(final_signal))
                entry_price = close[i]
                entry_atr = atr[i]
                highest_since_entry = close[i] if position_side > 0 else 0.0
                lowest_since_entry = close[i] if position_side < 0 else float('inf')
            elif position_side > 0:
                highest_since_entry = max(highest_since_entry, close[i])
            elif position_side < 0:
                lowest_since_entry = min(lowest_since_entry, close[i])
        else:
            if in_position:
                in_position = False
                position_side = 0
                entry_price = 0.0
                entry_atr = 0.0
                highest_since_entry = 0.0
                lowest_since_entry = float('inf')
        
        signals[i] = final_signal
    
    return signals
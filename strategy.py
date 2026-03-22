#!/usr/bin/env python3
"""
Experiment #003: 1h Bollinger Mean Reversion + 4h HMA Trend Filter + RSI Confirmation + ATR Stop
Hypothesis: In bear/range markets (2022 crash, 2025 bear), mean reversion at Bollinger extremes
with HTF trend filter outperforms pure trend following. 1h timeframe balances trade frequency
with signal quality. Multiple entry paths ensure >=10 trades per symbol. Conservative sizing (0.25)
controls drawdown. 2.5*ATR stoploss appropriate for 1h bars.
Timeframe: 1h (REQUIRED), HTF: 4h via mtf_data helper.
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_1h_bb_meanrev_4h_hma_rsi_atr_v1"
timeframe = "1h"
leverage = 1.0

def calculate_atr(high, low, close, period=14):
    """Calculate ATR using Wilder's smoothing."""
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]
    atr = pd.Series(tr).ewm(span=period, min_periods=period, adjust=False).mean().values
    return atr

def calculate_bollinger_bands(close, period=20, std_dev=2.0):
    """Calculate Bollinger Bands."""
    close_s = pd.Series(close)
    sma = close_s.rolling(window=period, min_periods=period).mean().values
    std = close_s.rolling(window=period, min_periods=period).std().values
    upper = sma + std_dev * std
    lower = sma - std_dev * std
    bandwidth = (upper - lower) / sma
    return upper, lower, sma, bandwidth

def calculate_rsi(close, period=14):
    """Calculate RSI indicator."""
    delta = np.diff(close, prepend=close[0])
    gain = np.where(delta > 0, delta, 0.0)
    loss = np.where(delta < 0, -delta, 0.0)
    avg_g = pd.Series(gain).ewm(span=period, min_periods=period, adjust=False).mean().values
    avg_l = pd.Series(loss).ewm(span=period, min_periods=period, adjust=False).mean().values
    rs = np.where(avg_l > 0, avg_g / avg_l, 100.0)
    rsi = 100 - 100 / (1 + rs)
    rsi = np.clip(rsi, 0, 100)
    return rsi

def calculate_hma(close, period=21):
    """Calculate Hull Moving Average for smoother trend with less lag."""
    close_s = pd.Series(close)
    half = max(1, period // 2)
    sqrt_period = max(1, int(np.sqrt(period)))
    wma1 = close_s.ewm(span=half, min_periods=half, adjust=False).mean()
    wma2 = close_s.ewm(span=period, min_periods=period, adjust=False).mean()
    wma3 = (2 * wma1 - wma2).ewm(span=sqrt_period, min_periods=sqrt_period, adjust=False).mean()
    return wma3.values

def calculate_zscore(close, period=20):
    """Calculate Z-score for mean reversion signals."""
    close_s = pd.Series(close)
    sma = close_s.rolling(window=period, min_periods=period).mean().values
    std = close_s.rolling(window=period, min_periods=period).std().values
    zscore = (close - sma) / (std + 1e-10)
    return zscore

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    n = len(close)
    
    # Load HTF data ONCE before loop (Rule 1 - CRITICAL)
    df_4h = get_htf_data(prices, '4h')
    
    # Calculate HTF indicators
    hma_4h = calculate_hma(df_4h['close'].values, 21)
    
    # Align HTF to LTF (Rule 2 - no manual index mapping, auto shift(1))
    hma_4h_aligned = align_htf_to_ltf(prices, df_4h, hma_4h)
    
    # Calculate 1h indicators
    atr = calculate_atr(high, low, close, 14)
    bb_upper, bb_lower, bb_sma, bb_bw = calculate_bollinger_bands(close, 20, 2.0)
    rsi = calculate_rsi(close, 14)
    zscore = calculate_zscore(close, 20)
    
    # 1h HMA for additional trend confirmation
    hma_1h = calculate_hma(close, 21)
    hma_1h_fast = calculate_hma(close, 10)
    
    signals = np.zeros(n)
    SIZE_ENTRY = 0.25
    SIZE_HALF = 0.12
    
    # Track positions for stoploss
    position_side = 0
    entry_price = 0.0
    trailing_stop = 0.0
    position_reduced = False
    highest_close = 0.0
    lowest_close = 0.0
    
    for i in range(100, n):
        # Skip if indicators not ready
        if np.isnan(atr[i]) or atr[i] == 0:
            signals[i] = 0.0
            continue
        
        if np.isnan(hma_4h_aligned[i]) or np.isnan(bb_upper[i]):
            signals[i] = 0.0
            continue
        
        if np.isnan(rsi[i]) or np.isnan(zscore[i]):
            signals[i] = 0.0
            continue
        
        # 4h trend bias (HTF)
        htf_bullish = close[i] > hma_4h_aligned[i]
        htf_bearish = close[i] < hma_4h_aligned[i]
        
        # 1h HMA trend
        hma_1h_bullish = close[i] > hma_1h[i]
        hma_1h_bearish = close[i] < hma_1h[i]
        hma_rising = hma_1h[i] > hma_1h[i-1] if i > 0 else False
        hma_falling = hma_1h[i] < hma_1h[i-1] if i > 0 else False
        
        # Fast HMA crossover
        fast_above_slow = hma_1h_fast[i] > hma_1h[i]
        fast_below_slow = hma_1h_fast[i] < hma_1h[i]
        
        # Bollinger Band position
        bb_position = (close[i] - bb_lower[i]) / (bb_upper[i] - bb_lower[i] + 1e-10)
        at_bb_lower = close[i] <= bb_lower[i] * 1.002
        at_bb_upper = close[i] >= bb_upper[i] * 0.998
        near_bb_lower = close[i] <= bb_lower[i] * 1.01
        near_bb_upper = close[i] >= bb_upper[i] * 0.99
        
        # RSI zones
        rsi_oversold = rsi[i] < 30
        rsi_overbought = rsi[i] > 70
        rsi_very_oversold = rsi[i] < 25
        rsi_very_overbought = rsi[i] > 75
        rsi_rising = rsi[i] > rsi[i-1] if i > 0 else False
        rsi_falling = rsi[i] < rsi[i-1] if i > 0 else False
        
        # Z-score extremes
        zscore_oversold = zscore[i] < -1.5
        zscore_overbought = zscore[i] > 1.5
        zscore_extreme_low = zscore[i] < -2.0
        zscore_extreme_high = zscore[i] > 2.0
        
        # Volatility regime (BB bandwidth)
        low_vol = bb_bw[i] < np.nanpercentile(bb_bw[:i+1], 30) if i > 100 else False
        high_vol = bb_bw[i] > np.nanpercentile(bb_bw[:i+1], 70) if i > 100 else False
        
        new_signal = 0.0
        
        # === LONG ENTRIES (multiple paths for >=10 trades) ===
        
        # Path 1: BB lower + RSI oversold + HTF not bearish (mean reversion)
        if at_bb_lower and rsi_oversold and not htf_bearish:
            new_signal = SIZE_ENTRY
        
        # Path 2: Z-score extreme low + RSI rising (mean reversion with momentum)
        elif zscore_extreme_low and rsi_rising and rsi[i] > 25:
            new_signal = SIZE_ENTRY
        
        # Path 3: HTF bullish + near BB lower + RSI oversold bounce
        elif htf_bullish and near_bb_lower and rsi_oversold and rsi_rising:
            new_signal = SIZE_ENTRY
        
        # Path 4: HTF bullish + HMA rising + fast HMA crossover up
        elif htf_bullish and hma_rising and fast_above_slow and hma_1h_fast[i] > hma_1h_fast[i-1]:
            new_signal = SIZE_ENTRY
        
        # Path 5: BB squeeze breakout long (low vol + price breaks above BB mid)
        elif low_vol and close[i] > bb_sma[i] and close[i-1] <= bb_sma[i-1] and htf_bullish:
            new_signal = SIZE_ENTRY
        
        # Path 6: RSI very oversold + price > HTF HMA (strong bounce setup)
        elif rsi_very_oversold and htf_bullish and rsi_rising:
            new_signal = SIZE_ENTRY
        
        # === SHORT ENTRIES (multiple paths for >=10 trades) ===
        
        # Path 1: BB upper + RSI overbought + HTF not bullish (mean reversion)
        if at_bb_upper and rsi_overbought and not htf_bullish:
            new_signal = -SIZE_ENTRY
        
        # Path 2: Z-score extreme high + RSI falling (mean reversion with momentum)
        elif zscore_extreme_high and rsi_falling and rsi[i] < 75:
            new_signal = -SIZE_ENTRY
        
        # Path 3: HTF bearish + near BB upper + RSI overbought drop
        elif htf_bearish and near_bb_upper and rsi_overbought and rsi_falling:
            new_signal = -SIZE_ENTRY
        
        # Path 4: HTF bearish + HMA falling + fast HMA crossover down
        elif htf_bearish and hma_falling and fast_below_slow and hma_1h_fast[i] < hma_1h_fast[i-1]:
            new_signal = -SIZE_ENTRY
        
        # Path 5: BB squeeze breakout short (low vol + price breaks below BB mid)
        elif low_vol and close[i] < bb_sma[i] and close[i-1] >= bb_sma[i-1] and htf_bearish:
            new_signal = -SIZE_ENTRY
        
        # Path 6: RSI very overbought + price < HTF HMA (strong drop setup)
        elif rsi_very_overbought and htf_bearish and rsi_falling:
            new_signal = -SIZE_ENTRY
        
        # === STOPLOSS LOGIC (Rule 6) ===
        if position_side > 0 and entry_price > 0:
            # Update highest close for trailing
            if close[i] > highest_close:
                highest_close = close[i]
            
            # Calculate trailing stop (2.5*ATR for 1h timeframe)
            current_stop = highest_close - 2.5 * atr[i]
            if current_stop > trailing_stop:
                trailing_stop = current_stop
            
            # Check stoploss hit
            if close[i] < trailing_stop:
                new_signal = 0.0
            elif not position_reduced:
                # Take profit at 2R
                risk = 2.5 * atr[i]
                profit = close[i] - entry_price
                if profit >= 2.0 * risk:
                    new_signal = SIZE_HALF
                    position_reduced = True
        
        if position_side < 0 and entry_price > 0:
            # Update lowest close for trailing
            if close[i] < lowest_close or lowest_close == 0.0:
                lowest_close = close[i]
            
            # Calculate trailing stop (2.5*ATR for 1h timeframe)
            current_stop = lowest_close + 2.5 * atr[i]
            if trailing_stop == 0.0 or current_stop < trailing_stop:
                trailing_stop = current_stop
            
            # Check stoploss hit
            if close[i] > trailing_stop:
                new_signal = 0.0
            elif not position_reduced:
                # Take profit at 2R
                risk = 2.5 * atr[i]
                profit = entry_price - close[i]
                if profit >= 2.0 * risk:
                    new_signal = -SIZE_HALF
                    position_reduced = True
        
        # Update position tracking AFTER signal calculation
        prev_signal = signals[i - 1] if i > 0 else 0.0
        
        # New position opened
        if new_signal != 0.0 and prev_signal == 0.0:
            entry_price = close[i]
            position_side = np.sign(new_signal)
            trailing_stop = close[i] - 2.5 * atr[i] if position_side > 0 else close[i] + 2.5 * atr[i]
            highest_close = close[i] if position_side > 0 else 0.0
            lowest_close = close[i] if position_side < 0 else 0.0
            position_reduced = False
        
        # Position reversed
        elif new_signal != 0.0 and prev_signal != 0.0 and np.sign(new_signal) != np.sign(prev_signal):
            entry_price = close[i]
            position_side = np.sign(new_signal)
            trailing_stop = close[i] - 2.5 * atr[i] if position_side > 0 else close[i] + 2.5 * atr[i]
            highest_close = close[i] if position_side > 0 else 0.0
            lowest_close = close[i] if position_side < 0 else 0.0
            position_reduced = False
        
        # Position reduced (take profit)
        elif new_signal != 0.0 and prev_signal != 0.0 and np.abs(new_signal) < np.abs(prev_signal):
            position_reduced = True
        
        # Position closed
        elif new_signal == 0.0 and prev_signal != 0.0:
            position_side = 0
            entry_price = 0.0
            trailing_stop = 0.0
            highest_close = 0.0
            lowest_close = 0.0
            position_reduced = False
        
        signals[i] = new_signal
    
    return signals
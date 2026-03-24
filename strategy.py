#!/usr/bin/env python3
"""
Experiment #1443: 1d Primary + 1w HTF — Simplified Donchian + HMA Trend + Volume

Hypothesis: Previous strategy (#1442) failed because entry conditions were too complex
(multiple regime filters + CRSI + Donchian all needing to align). This version simplifies:
1. Remove Choppiness Index (causes whipsaw, inconsistent across symbols)
2. Keep 1w HMA as single macro trend filter (proven to work)
3. Donchian(20) breakout for entries (simpler than dual Donchian)
4. Add volume confirmation on breakouts (filters false breakouts)
5. CRSI only for counter-trend entries in strong trends (simplified thresholds)
6. ATR(14) trailing stop 2.5x for risk management

Why this should work better:
- Fewer AND conditions = more trades generated (critical for passing trade count)
- Volume filter reduces false breakouts without over-complicating
- 1w HMA is the strongest single filter for macro direction
- Simpler logic = more consistent across BTC/ETH/SOL (all must have Sharpe>0)

Design:
1. 1w HMA(21) = macro trend (call ONCE before loop, align properly)
2. Donchian(20) breakout = entry trigger
3. Volume > SMA(volume, 20) * 1.2 = breakout confirmation
4. Long: price > 1w_HMA + Donchian breakout + volume confirm
5. Short: price < 1w_HMA + Donchian breakdown + volume confirm
6. CRSI < 20 for long adds in strong uptrend (pullback entry)
7. ATR(14) trailing stop 2.5x
8. Position size 0.25 (discrete: 0.0, ±0.25)

Target: 30-60 trades/year, Sharpe > 0.618, trades >= 30 train, >= 3 test ALL symbols
Timeframe: 1d
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_1d_donchian_hma_vol_1w_atr_v2"
timeframe = "1d"
leverage = 1.0

def calculate_hma(close, period=21):
    """Hull Moving Average - faster response than EMA, less lag"""
    n = len(close)
    if n < period:
        return np.full(n, np.nan)
    
    half = period // 2
    sqrt_period = int(np.sqrt(period))
    
    def wma(series, span):
        weights = np.arange(1, span + 1, dtype=np.float64)
        result = np.full(len(series), np.nan)
        for i in range(span - 1, len(series)):
            window = series[i - span + 1:i + 1]
            if not np.any(np.isnan(window)):
                result[i] = np.sum(window * weights) / np.sum(weights)
        return result
    
    wma_half = wma(close, half)
    wma_full = wma(close, period)
    
    hma = np.full(n, np.nan)
    for i in range(period - 1, n):
        if not np.isnan(wma_half[i]) and not np.isnan(wma_full[i]):
            diff = 2.0 * wma_half[i] - wma_full[i]
            if i >= sqrt_period - 1:
                diff_vals = []
                for j in range(i - sqrt_period + 1, i + 1):
                    if j >= period - 1 and not np.isnan(2.0 * wma_half[j] - wma_full[j]):
                        diff_vals.append(2.0 * wma_half[j] - wma_full[j])
                if len(diff_vals) == sqrt_period:
                    weights = np.arange(1, sqrt_period + 1, dtype=np.float64)
                    hma[i] = np.sum(np.array(diff_vals) * weights) / np.sum(weights)
    
    return hma

def calculate_rsi(close, period=14):
    """Relative Strength Index"""
    n = len(close)
    if n < period + 1:
        return np.full(n, np.nan)
    
    delta = np.diff(close, prepend=close[0])
    gain = np.where(delta > 0, delta, 0.0)
    loss = np.where(delta < 0, -delta, 0.0)
    
    gain_smooth = pd.Series(gain).ewm(span=period, min_periods=period, adjust=False).mean().values
    loss_smooth = pd.Series(loss).ewm(span=period, min_periods=period, adjust=False).mean().values
    
    rsi = np.full(n, np.nan)
    mask = loss_smooth > 1e-10
    rsi[mask] = 100.0 - (100.0 / (1.0 + gain_smooth[mask] / loss_smooth[mask]))
    rsi[loss_smooth <= 1e-10] = 100.0
    rsi[:period] = np.nan
    
    return rsi

def calculate_crsi_simple(close, rsi_period=3, rank_period=50):
    """Simplified Connors RSI for faster computation"""
    n = len(close)
    if n < rank_period + rsi_period:
        return np.full(n, np.nan)
    
    crsi = np.full(n, np.nan)
    
    # RSI(3)
    delta = np.diff(close, prepend=close[0])
    gain = np.where(delta > 0, delta, 0.0)
    loss = np.where(delta < 0, -delta, 0.0)
    
    gain_smooth = pd.Series(gain).ewm(span=rsi_period, min_periods=rsi_period, adjust=False).mean().values
    loss_smooth = pd.Series(loss).ewm(span=rsi_period, min_periods=rsi_period, adjust=False).mean().values
    
    rsi_short = np.full(n, np.nan)
    mask = loss_smooth > 1e-10
    rsi_short[mask] = 100.0 - (100.0 / (1.0 + gain_smooth[mask] / loss_smooth[mask]))
    rsi_short[loss_smooth <= 1e-10] = 100.0
    
    # Streak RSI (simplified)
    streak_rsi = np.full(n, np.nan)
    for i in range(2, n):
        streak = 0
        if close[i] > close[i-1]:
            j = i
            while j > 0 and close[j] > close[j-1]:
                streak += 1
                j -= 1
        elif close[i] < close[i-1]:
            j = i
            while j > 0 and close[j] < close[j-1]:
                streak -= 1
                j -= 1
        streak_rsi[i] = 50.0 + streak * 25.0
        streak_rsi[i] = np.clip(streak_rsi[i], 0.0, 100.0)
    
    # Percent Rank (simplified to 50 period for speed)
    percent_rank = np.full(n, np.nan)
    for i in range(rank_period, n):
        returns = np.diff(close[i-rank_period+1:i+1])
        if len(returns) > 0 and not np.any(np.isnan(returns)):
            current_return = returns[-1]
            count_below = np.sum(returns[:-1] < current_return)
            percent_rank[i] = 100.0 * count_below / (len(returns) - 1) if len(returns) > 1 else 50.0
    
    # Combine
    for i in range(rank_period, n):
        if not np.isnan(rsi_short[i]) and not np.isnan(streak_rsi[i]) and not np.isnan(percent_rank[i]):
            crsi[i] = (rsi_short[i] + streak_rsi[i] + percent_rank[i]) / 3.0
    
    return crsi

def calculate_atr(high, low, close, period=14):
    """Average True Range - for stoploss sizing"""
    n = len(close)
    if n < period + 1:
        return np.full(n, np.nan)
    
    tr = np.zeros(n)
    tr[0] = high[0] - low[0]
    for i in range(1, n):
        tr[i] = max(high[i] - low[i], abs(high[i] - close[i-1]), abs(low[i] - close[i-1]))
    
    atr = pd.Series(tr).ewm(span=period, min_periods=period, adjust=False).mean().values
    return atr

def calculate_donchian(high, low, period=20):
    """Donchian Channel - breakout levels for entry trigger"""
    n = len(high)
    if n < period:
        return np.full(n, np.nan), np.full(n, np.nan)
    
    upper = np.full(n, np.nan)
    lower = np.full(n, np.nan)
    
    for i in range(period - 1, n):
        upper[i] = np.nanmax(high[i-period+1:i+1])
        lower[i] = np.nanmin(low[i-period+1:i+1])
    
    return upper, lower

def calculate_volume_sma(volume, period=20):
    """Simple moving average of volume"""
    n = len(volume)
    if n < period:
        return np.full(n, np.nan)
    
    vol_sma = pd.Series(volume).rolling(window=period, min_periods=period).mean().values
    return vol_sma

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    volume = prices["volume"].values
    n = len(close)
    
    # Load HTF data ONCE before loop (Rule 1 - CRITICAL)
    df_1w = get_htf_data(prices, '1w')
    
    # Calculate and align 1w HMA for macro trend
    hma_1w_raw = calculate_hma(df_1w['close'].values, period=21)
    hma_1w_aligned = align_htf_to_ltf(prices, df_1w, hma_1w_raw)
    
    # Calculate primary (1d) indicators
    donchian_20_upper, donchian_20_lower = calculate_donchian(high, low, period=20)
    donchian_55_upper, donchian_55_lower = calculate_donchian(high, low, period=55)
    crsi = calculate_crsi_simple(close, rsi_period=3, rank_period=50)
    atr = calculate_atr(high, low, close, period=14)
    vol_sma = calculate_volume_sma(volume, period=20)
    
    # Also calculate 1d HMA for additional trend confirmation
    hma_1d = calculate_hma(close, period=21)
    
    signals = np.zeros(n)
    BASE_SIZE = 0.25
    
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
            if in_position:
                in_position = False
                position_side = 0
            continue
        if np.isnan(donchian_20_upper[i]) or np.isnan(hma_1w_aligned[i]):
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        if np.isnan(hma_1d[i]) or np.isnan(vol_sma[i]):
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        
        # === MACRO TREND (1w HMA) ===
        macro_bull = close[i] > hma_1w_aligned[i]
        macro_bear = close[i] < hma_1w_aligned[i]
        
        # === VOLUME CONFIRMATION ===
        volume_confirm = volume[i] > vol_sma[i] * 1.15  # 15% above average
        
        # === DONCHIAN BREAKOUT ===
        breakout_long = close[i] > donchian_20_upper[i-1] if i > 0 else False
        breakout_short = close[i] < donchian_20_lower[i-1] if i > 0 else False
        
        # === 1D HMA CONFIRMATION ===
        hma_1d_bull = close[i] > hma_1d[i]
        hma_1d_bear = close[i] < hma_1d[i]
        
        # === DESIRED SIGNAL - SIMPLIFIED LOGIC ===
        desired_signal = 0.0
        
        # LONG ENTRIES
        # Path 1: Macro bull + Donchian breakout + volume confirm (primary)
        if macro_bull and breakout_long and volume_confirm:
            desired_signal = BASE_SIZE
        # Path 2: Macro bull + CRSI oversold (pullback entry in uptrend)
        elif macro_bull and not np.isnan(crsi[i]) and crsi[i] < 25.0:
            desired_signal = BASE_SIZE * 0.5
        # Path 3: Both HMA aligned bull + Donchian breakout (stronger signal)
        elif macro_bull and hma_1d_bull and breakout_long:
            desired_signal = BASE_SIZE
        
        # SHORT ENTRIES
        # Path 1: Macro bear + Donchian breakdown + volume confirm (primary)
        elif macro_bear and breakout_short and volume_confirm:
            desired_signal = -BASE_SIZE
        # Path 2: Macro bear + CRSI overbought (pullback entry in downtrend)
        elif macro_bear and not np.isnan(crsi[i]) and crsi[i] > 75.0:
            desired_signal = -BASE_SIZE * 0.5
        # Path 3: Both HMA aligned bear + Donchian breakdown (stronger signal)
        elif macro_bear and hma_1d_bear and breakout_short:
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
        
        # === DISCRETIZE SIGNAL VALUES ===
        if abs(desired_signal) >= BASE_SIZE * 0.4:
            if desired_signal > 0:
                final_signal = BASE_SIZE
            else:
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
                # Flip position
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
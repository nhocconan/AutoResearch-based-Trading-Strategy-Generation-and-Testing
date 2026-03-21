#!/usr/bin/env python3
"""
Experiment #336: 1d Dual Momentum + Weekly HMA Trend + Fisher Transform Entries + Volatility Scaling
Hypothesis: Daily timeframe needs fewer but higher-quality trades. Combining dual momentum
(absolute + relative) with weekly trend filter and Fisher Transform for precise entry timing.
Fisher Transform catches reversals in bear markets (2022, 2025) better than RSI. Weekly HMA(21)
provides macro bias to avoid counter-trend trades. Volatility-based position sizing reduces
exposure during high-vol regimes (2022 crash). Target: 15-30 trades over 4 years, Sharpe>0.5.
Timeframe: 1d (REQUIRED), HTF: 1w for trend bias via mtf_data helper.
Key innovation: Fisher Transform (-1.2/+1.2 thresholds) + KAMA adaptive trend + vol scaling.
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_1d_dual_momentum_weekly_fisher_kama_vol_v1"
timeframe = "1d"
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

def calculate_hma(close, period=21):
    """Calculate Hull Moving Average for faster trend response."""
    close_s = pd.Series(close)
    half = max(1, period // 2)
    sqrt_period = max(1, int(np.sqrt(period)))
    wma1 = close_s.ewm(span=half, min_periods=half, adjust=False).mean()
    wma2 = close_s.ewm(span=period, min_periods=period, adjust=False).mean()
    wma3 = (2 * wma1 - wma2).ewm(span=sqrt_period, min_periods=sqrt_period, adjust=False).mean()
    return wma3.values

def calculate_kama(close, period=10, fast=2, slow=30):
    """Calculate Kaufman Adaptive Moving Average."""
    close_s = pd.Series(close)
    change = np.abs(close_s - close_s.shift(period))
    volatility = pd.Series(np.abs(close_s - close_s.shift(1))).rolling(window=period, min_periods=period).sum()
    er = change / volatility.replace(0, np.inf)
    er = er.fillna(0)
    fast_sc = 2 / (fast + 1)
    slow_sc = 2 / (slow + 1)
    sc = (er * (fast_sc - slow_sc) + slow_sc) ** 2
    kama = pd.Series(index=close_s.index, dtype=float)
    kama.iloc[period-1] = close_s.iloc[period-1]
    for i in range(period, len(close_s)):
        kama.iloc[i] = kama.iloc[i-1] + sc.iloc[i] * (close_s.iloc[i] - kama.iloc[i-1])
    return kama.values

def calculate_fisher_transform(high, low, period=9):
    """Calculate Ehlers Fisher Transform for reversal detection."""
    hl2 = (high + low) / 2
    highest = pd.Series(hl2).rolling(window=period, min_periods=period).max().values
    lowest = pd.Series(hl2).rolling(window=period, min_periods=period).min().values
    range_val = highest - lowest
    range_val = np.where(range_val < 0.001, 0.001, range_val)
    normalized = (hl2 - lowest) / range_val
    normalized = np.clip(normalized, 0.001, 0.999)
    fisher_input = 0.66 * ((normalized - 0.5) / 0.5) + 0.67 * np.roll(((normalized - 0.5) / 0.5), 1)
    fisher_input[0] = 0
    fisher = 0.5 * np.log((1 + fisher_input) / (1 - fisher_input + 0.001))
    fisher_signal = np.roll(fisher, 1)
    fisher_signal[0] = fisher[0]
    return fisher, fisher_signal

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

def calculate_sma(close, period=200):
    """Calculate Simple Moving Average."""
    return pd.Series(close).rolling(window=period, min_periods=period).mean().values

def calculate_momentum(close, period=10):
    """Calculate price momentum (ROC)."""
    return (close - np.roll(close, period)) / np.roll(close, period) * 100

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    n = len(close)
    
    # Load HTF data ONCE before loop (Rule 1)
    df_1w = get_htf_data(prices, '1w')
    
    # Calculate HTF indicators
    hma_1w = calculate_hma(df_1w['close'].values, 21)
    
    # Align HTF to LTF (Rule 2 - no manual index mapping)
    hma_1w_aligned = align_htf_to_ltf(prices, df_1w, hma_1w)
    
    # Calculate 1d indicators
    atr = calculate_atr(high, low, close, 14)
    rsi = calculate_rsi(close, 14)
    fisher, fisher_signal = calculate_fisher_transform(high, low, 9)
    kama = calculate_kama(close, 10)
    sma200 = calculate_sma(close, 200)
    momentum = calculate_momentum(close, 10)
    
    # Volatility regime (ATR ratio)
    atr_ma = pd.Series(atr).rolling(window=50, min_periods=50).mean().values
    vol_ratio = atr / np.where(atr_ma > 0, atr_ma, atr)
    
    signals = np.zeros(n)
    SIZE_BASE = 0.30
    SIZE_REDUCED = 0.15
    
    # Track positions for stoploss
    position_side = 0
    entry_price = 0.0
    trailing_stop = 0.0
    position_reduced = False
    highest_close = 0.0
    lowest_close = 0.0
    
    for i in range(250, n):  # Start after 250 bars for indicators
        # Skip if indicators not ready
        if np.isnan(atr[i]) or np.isnan(rsi[i]) or np.isnan(fisher[i]) or np.isnan(kama[i]):
            signals[i] = 0.0
            continue
        
        # Weekly macro trend bias
        weekly_bullish = not np.isnan(hma_1w_aligned[i]) and close[i] > hma_1w_aligned[i]
        weekly_bearish = not np.isnan(hma_1w_aligned[i]) and close[i] < hma_1w_aligned[i]
        
        # Absolute momentum (price vs SMA200)
        abs_mom_long = not np.isnan(sma200[i]) and close[i] > sma200[i]
        abs_mom_short = not np.isnan(sma200[i]) and close[i] < sma200[i]
        
        # Relative momentum (10-period ROC)
        rel_mom_long = momentum[i] > 0
        rel_mom_short = momentum[i] < 0
        
        # KAMA adaptive trend
        kama_bullish = close[i] > kama[i]
        kama_bearish = close[i] < kama[i]
        
        # Fisher Transform entry signals (LOOSE thresholds for more trades)
        fisher_long = fisher[i] > -1.2 and fisher_signal[i] <= -1.2  # Cross above -1.2
        fisher_short = fisher[i] < 1.2 and fisher_signal[i] >= 1.2  # Cross below +1.2
        
        # RSI confirmation (not extreme, just directional)
        rsi_ok_long = rsi[i] > 40
        rsi_ok_short = rsi[i] < 60
        
        # Volatility-based position sizing
        vol_scale = 1.0
        if vol_ratio[i] > 1.5:
            vol_scale = 0.67  # Reduce size in high vol
        elif vol_ratio[i] < 0.7:
            vol_scale = 1.2  # Increase size in low vol
        
        SIZE_ENTRY = min(SIZE_BASE * vol_scale, 0.35)
        SIZE_HALF = SIZE_ENTRY / 2
        
        new_signal = 0.0
        
        # === LONG ENTRIES ===
        # Primary: Weekly bullish + Fisher long + KAMA bullish
        if weekly_bullish and fisher_long and kama_bullish:
            new_signal = SIZE_ENTRY
        # Secondary: Absolute momentum + Relative momentum + RSI ok
        elif abs_mom_long and rel_mom_long and rsi_ok_long and weekly_bullish:
            new_signal = SIZE_ENTRY
        # Tertiary: KAMA cross + momentum (looser)
        elif kama_bullish and rel_mom_long and rsi[i] > 45:
            new_signal = SIZE_ENTRY
        
        # === SHORT ENTRIES ===
        # Primary: Weekly bearish + Fisher short + KAMA bearish
        if weekly_bearish and fisher_short and kama_bearish:
            new_signal = -SIZE_ENTRY
        # Secondary: Absolute momentum + Relative momentum + RSI ok
        elif abs_mom_short and rel_mom_short and rsi_ok_short and weekly_bearish:
            new_signal = -SIZE_ENTRY
        # Tertiary: KAMA cross + momentum (looser)
        elif kama_bearish and rel_mom_short and rsi[i] < 55:
            new_signal = -SIZE_ENTRY
        
        # === STOPLOSS LOGIC (Rule 6) ===
        if position_side > 0 and entry_price > 0:
            # Update highest close for trailing
            if close[i] > highest_close:
                highest_close = close[i]
            
            # Calculate trailing stop (3*ATR from highest)
            current_stop = highest_close - 3.0 * atr[i]
            if current_stop > trailing_stop:
                trailing_stop = current_stop
            
            # Check stoploss hit
            if close[i] < trailing_stop:
                new_signal = 0.0
            elif not position_reduced:
                # Take profit at 2.5R
                risk = 3.0 * atr[i]
                profit = close[i] - entry_price
                if profit >= 2.5 * risk:
                    new_signal = SIZE_HALF
                    position_reduced = True
        
        if position_side < 0 and entry_price > 0:
            # Update lowest close for trailing
            if close[i] < lowest_close or lowest_close == 0.0:
                lowest_close = close[i]
            
            # Calculate trailing stop (3*ATR from lowest)
            current_stop = lowest_close + 3.0 * atr[i]
            if trailing_stop == 0.0 or current_stop < trailing_stop:
                trailing_stop = current_stop
            
            # Check stoploss hit
            if close[i] > trailing_stop:
                new_signal = 0.0
            elif not position_reduced:
                # Take profit at 2.5R
                risk = 3.0 * atr[i]
                profit = entry_price - close[i]
                if profit >= 2.5 * risk:
                    new_signal = -SIZE_HALF
                    position_reduced = True
        
        # Update position tracking AFTER signal calculation
        prev_signal = signals[i-1] if i > 0 else 0.0
        
        # New position opened
        if new_signal != 0.0 and prev_signal == 0.0:
            entry_price = close[i]
            position_side = np.sign(new_signal)
            trailing_stop = close[i] - 3.0 * atr[i] if position_side > 0 else close[i] + 3.0 * atr[i]
            highest_close = close[i] if position_side > 0 else 0.0
            lowest_close = close[i] if position_side < 0 else 0.0
            position_reduced = False
        
        # Position reversed
        elif new_signal != 0.0 and prev_signal != 0.0 and np.sign(new_signal) != np.sign(prev_signal):
            entry_price = close[i]
            position_side = np.sign(new_signal)
            trailing_stop = close[i] - 3.0 * atr[i] if position_side > 0 else close[i] + 3.0 * atr[i]
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
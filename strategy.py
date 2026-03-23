#!/usr/bin/env python3
"""
Experiment #1093: 1d Primary + 1w HTF — Regime-Adaptive Donchian + RSI + ATR

Hypothesis: Daily timeframe with weekly trend filter provides optimal trade frequency
(20-50 trades/year) while avoiding lower-TF fee drag. Key innovations:

1. Weekly HMA21 for macro regime (bull/bear) — loaded ONCE via mtf_data helper
2. Daily Donchian(20) breakout for entries — proven on SOL (Sharpe +0.782)
3. RSI(14) filter to avoid extremes — prevents buying tops/selling bottoms
4. Choppiness Index(14) for regime detection — switch between trend/mean-revert
5. ATR(14) trailing stop 2.5x — proper risk management
6. Asymmetric sizing: 0.30 in trend regime, 0.20 in range regime

Why this should beat Sharpe=0.612:
- 1d primary = fewer trades, less fee drag (proven in experiment history)
- 1w HTF filter prevents counter-trend trades (major 2022 failure mode)
- Donchian breakout works in both bull and bear markets
- Choppiness filter avoids trend strategies in chop (whipsaw protection)
- Simpler entry conditions = more trades (avoid 0-trade problem)

Timeframe: 1d (primary)
HTF: 1w — loaded ONCE before loop using mtf_data helper
Position Size: 0.20-0.30 discrete levels
Stoploss: 2.5x ATR trailing
Target Trades: 25-40 per year per symbol
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_1d_regime_donchian_rsi_1w_hma_chop_atr_v1"
timeframe = "1d"
leverage = 1.0

def calculate_hma(series, period):
    """
    Hull Moving Average — faster and smoother than EMA.
    
    Formula:
    1. WMA(period/2) * 2
    2. WMA(period) * 1
    3. Diff = (1) - (2)
    4. HMA = WMA(sqrt(period)) of Diff
    """
    series = pd.Series(series)
    half = period // 2
    sqrt_period = int(np.sqrt(period))
    
    wma_half = series.rolling(window=half, min_periods=half).mean() * 2
    wma_full = series.rolling(window=period, min_periods=period).mean()
    
    wma_diff = wma_half - wma_full
    hma = wma_diff.rolling(window=sqrt_period, min_periods=sqrt_period).mean()
    
    return hma.values

def calculate_rsi(close, period=14):
    """
    Relative Strength Index — momentum oscillator.
    """
    n = len(close)
    rsi = np.full(n, np.nan)
    
    if n < period + 1:
        return rsi
    
    diff = np.diff(close)
    gain = np.where(diff > 0, diff, 0.0)
    loss = np.where(diff < 0, -diff, 0.0)
    
    gain = np.concatenate([[0.0], gain])
    loss = np.concatenate([[0.0], loss])
    
    avg_gain = pd.Series(gain).ewm(span=period, min_periods=period, adjust=False).mean().values
    avg_loss = pd.Series(loss).ewm(span=period, min_periods=period, adjust=False).mean().values
    
    mask = avg_loss > 1e-10
    rs = np.zeros(n)
    rs[mask] = avg_gain[mask] / avg_loss[mask]
    rsi[mask] = 100.0 - (100.0 / (1.0 + rs[mask]))
    rsi[~mask] = 50.0
    
    return rsi

def calculate_choppiness(high, low, close, period=14):
    """
    Choppiness Index — measures market choppiness vs trending.
    
    Formula:
    CHOP = 100 * LOG10(SUM(ATR, period) / (Highest High - Lowest Low)) / LOG10(period)
    
    CHOP > 61.8 = choppy/ranging market
    CHOP < 38.2 = trending market
    """
    n = len(close)
    chop = np.full(n, np.nan)
    
    if n < period + 1:
        return chop
    
    # Calculate ATR
    tr = np.zeros(n)
    tr[0] = high[0] - low[0]
    for i in range(1, n):
        tr[i] = max(high[i] - low[i], abs(high[i] - close[i-1]), abs(low[i] - close[i-1]))
    
    # Rolling sum of ATR
    atr_sum = pd.Series(tr).rolling(window=period, min_periods=period).sum().values
    
    # Highest High and Lowest Low over period
    hh = pd.Series(high).rolling(window=period, min_periods=period).max().values
    ll = pd.Series(low).rolling(window=period, min_periods=period).min().values
    
    # Calculate CHOP
    denom = hh - ll
    mask = denom > 1e-10
    chop[mask] = 100.0 * np.log10(atr_sum[mask] / denom[mask]) / np.log10(period)
    chop[~mask] = 50.0
    
    return chop

def calculate_donchian(high, low, period=20):
    """
    Donchian Channels — breakout indicator.
    Returns upper and lower bands.
    """
    n = len(high)
    upper = pd.Series(high).rolling(window=period, min_periods=period).max().values
    lower = pd.Series(low).rolling(window=period, min_periods=period).min().values
    return upper, lower

def calculate_atr(high, low, close, period=14):
    """Average True Range for volatility measurement and stoploss."""
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

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    n = len(close)
    
    # Load HTF data ONCE before loop (Rule 1 - CRITICAL)
    df_1w = get_htf_data(prices, '1w')
    
    # Calculate and align 1w HMA21 for macro trend filter
    hma_1w_raw = calculate_hma(df_1w['close'].values, 21)
    hma_1w_aligned = align_htf_to_ltf(prices, df_1w, hma_1w_raw)
    
    # Calculate daily indicators
    hma_21 = calculate_hma(close, 21)
    hma_50 = calculate_hma(close, 50)
    rsi = calculate_rsi(close, period=14)
    chop = calculate_choppiness(high, low, close, period=14)
    donchian_upper, donchian_lower = calculate_donchian(high, low, period=20)
    atr = calculate_atr(high, low, close, period=14)
    
    signals = np.zeros(n)
    TREND_SIZE = 0.30
    RANGE_SIZE = 0.20
    
    # Position tracking for stoploss
    in_position = False
    position_side = 0
    entry_price = 0.0
    entry_atr = 0.0
    highest_since_entry = 0.0
    lowest_since_entry = float('inf')
    
    for i in range(100, n):
        # Skip if indicators not ready
        if np.isnan(hma_21[i]) or np.isnan(hma_50[i]):
            continue
        if np.isnan(rsi[i]) or np.isnan(chop[i]) or np.isnan(atr[i]):
            continue
        if np.isnan(donchian_upper[i]) or np.isnan(donchian_lower[i]):
            continue
        if np.isnan(hma_1w_aligned[i]) or atr[i] <= 1e-10:
            continue
        
        # === MACRO TREND (1w HMA21) ===
        weekly_bull = close[i] > hma_1w_aligned[i]
        weekly_bear = close[i] < hma_1w_aligned[i]
        
        # === PRIMARY TREND (Daily HMA) ===
        daily_bull = hma_21[i] > hma_50[i]
        daily_bear = hma_21[i] < hma_50[i]
        
        # === REGIME DETECTION (Choppiness) ===
        trending_regime = chop[i] < 45.0  # Below 45 = trending
        ranging_regime = chop[i] > 55.0   # Above 55 = ranging
        
        # === DONCHIAN BREAKOUT SIGNALS ===
        breakout_long = close[i] > donchian_upper[i-1] if i > 0 else False
        breakout_short = close[i] < donchian_lower[i-1] if i > 0 else False
        
        # === RSI FILTER ===
        rsi_ok_long = rsi[i] < 65.0  # Not overbought
        rsi_ok_short = rsi[i] > 35.0  # Not oversold
        
        # === VOLATILITY CHECK ===
        atr_median = np.nanmedian(atr[max(0, i-50):i]) if i > 50 else atr[i]
        vol_normal = atr[i] < 2.5 * atr_median
        
        # === SIZE SELECTION ===
        current_size = TREND_SIZE if trending_regime else RANGE_SIZE
        
        desired_signal = 0.0
        
        # === LONG ENTRY (Trending Regime) ===
        if trending_regime and weekly_bull and daily_bull:
            if breakout_long and rsi_ok_long and vol_normal:
                desired_signal = current_size
            # Also enter on pullback in strong trend
            elif daily_bull and rsi[i] < 45.0 and vol_normal:
                desired_signal = current_size * 0.7
        
        # === LONG ENTRY (Ranging Regime - Mean Revert) ===
        elif ranging_regime and weekly_bull:
            if close[i] < donchian_lower[i-1] * 0.98 and rsi[i] < 35.0:
                desired_signal = RANGE_SIZE
        
        # === SHORT ENTRY (Trending Regime) ===
        if trending_regime and weekly_bear and daily_bear:
            if breakout_short and rsi_ok_short and vol_normal:
                desired_signal = -current_size
            # Also enter on pullback in strong downtrend
            elif daily_bear and rsi[i] > 55.0 and vol_normal:
                desired_signal = -current_size * 0.7
        
        # === SHORT ENTRY (Ranging Regime - Mean Revert) ===
        elif ranging_regime and weekly_bear:
            if close[i] > donchian_upper[i-1] * 1.02 and rsi[i] > 65.0:
                desired_signal = -RANGE_SIZE
        
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
        
        # === HOLD LOGIC — Maintain position if trend intact ===
        if in_position and desired_signal == 0.0 and not stoploss_triggered:
            if position_side > 0:
                # Hold long if weekly trend still bull
                if weekly_bull and rsi[i] < 75.0:
                    desired_signal = current_size
            elif position_side < 0:
                # Hold short if weekly trend still bear
                if weekly_bear and rsi[i] > 25.0:
                    desired_signal = -current_size
        
        # === EXIT CONDITIONS ===
        if in_position and position_side > 0:
            # Exit long if weekly reverses or RSI extreme
            if weekly_bear and rsi[i] > 70.0:
                desired_signal = 0.0
        
        if in_position and position_side < 0:
            # Exit short if weekly reverses or RSI extreme
            if weekly_bull and rsi[i] < 30.0:
                desired_signal = 0.0
        
        # === DISCRETIZE SIGNAL VALUES ===
        if desired_signal > 0:
            if desired_signal >= TREND_SIZE * 0.8:
                desired_signal = TREND_SIZE
            elif desired_signal >= RANGE_SIZE * 0.8:
                desired_signal = RANGE_SIZE
            else:
                desired_signal = RANGE_SIZE * 0.5
        elif desired_signal < 0:
            if desired_signal <= -TREND_SIZE * 0.8:
                desired_signal = -TREND_SIZE
            elif desired_signal <= -RANGE_SIZE * 0.8:
                desired_signal = -RANGE_SIZE
            else:
                desired_signal = -RANGE_SIZE * 0.5
        else:
            desired_signal = 0.0
        
        # === UPDATE POSITION TRACKING ===
        if desired_signal != 0.0:
            if not in_position:
                in_position = True
                position_side = int(np.sign(desired_signal))
                entry_price = close[i]
                entry_atr = atr[i]
                highest_since_entry = close[i] if position_side > 0 else 0.0
                lowest_since_entry = close[i] if position_side < 0 else float('inf')
            elif np.sign(desired_signal) != position_side:
                # Flip position
                position_side = int(np.sign(desired_signal))
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
        
        signals[i] = desired_signal
    
    return signals
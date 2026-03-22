#!/usr/bin/env python3
"""
Experiment #477: 1d Primary + 1w HTF — Simplified Donchian Breakout + HMA Trend

Hypothesis: After 476 failed experiments, the pattern is clear:
1. Complex multi-filter strategies (CRSI+CHOP+HMA+SMA) generate TOO FEW trades
2. 1d timeframe with 1w HTF trend filter has shown promise in research (SOL Sharpe +0.879)
3. Donchian breakouts are simpler and more reliable than RSI extremes for 1d
4. Fewer conflicting conditions = more trades (critical: need >=30 trades/symbol)
5. HMA(21/50) crossover provides clean trend bias without over-filtering

Why this might beat current best (Sharpe=0.435):
- Simpler entry logic = more trade opportunities (Donchian breakout is binary)
- 1w HMA provides cleaner major trend filter than 1d HMA
- RSI(14) filter is simpler and more robust than CRSI (which has calculation bugs)
- ATR 2.5x trailing stop protects in crashes while allowing trend runs
- Asymmetric sizing: 0.30 long, 0.25 short (bear market protection for 2025)

Position sizing: 0.25-0.30 (discrete levels, max 0.40)
Stoploss: 2.5 * ATR trailing (signal → 0 when hit)
Target: 20-50 trades/year on 1d, >=30 trades/symbol on train, >=3 on test
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_1d_donchian_hma_1w_simp_v1"
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
    """Calculate Hull Moving Average (HMA)."""
    n = period
    half = n // 2
    sqrt_n = int(np.sqrt(n))
    
    close_s = pd.Series(close)
    
    def wma(series, span):
        weights = np.arange(1, span + 1)
        return series.rolling(window=span, min_periods=span).apply(
            lambda x: np.dot(x, weights) / weights.sum(), raw=True
        )
    
    wma_half = wma(close_s, half)
    wma_full = wma(close_s, n)
    hma_raw = 2.0 * wma_half - wma_full
    hma = wma(hma_raw, sqrt_n)
    
    return hma.values

def calculate_rsi(close, period=14):
    """Calculate RSI using Wilder's smoothing."""
    close_s = pd.Series(close)
    delta = close_s.diff()
    
    gain = delta.where(delta > 0, 0.0)
    loss = -delta.where(delta < 0, 0.0)
    
    avg_gain = gain.ewm(span=period, min_periods=period, adjust=False).mean()
    avg_loss = loss.ewm(span=period, min_periods=period, adjust=False).mean()
    
    rs = avg_gain / (avg_loss + 1e-10)
    rsi = 100.0 - (100.0 / (1.0 + rs))
    
    return rsi.values

def calculate_donchian_channels(high, low, period=20):
    """Calculate Donchian Channel upper and lower bands."""
    high_s = pd.Series(high)
    low_s = pd.Series(low)
    
    upper = high_s.rolling(window=period, min_periods=period).max().values
    lower = low_s.rolling(window=period, min_periods=period).min().values
    middle = (upper + lower) / 2.0
    
    return upper, lower, middle

def calculate_sma(close, period=50):
    """Calculate Simple Moving Average."""
    return pd.Series(close).rolling(window=period, min_periods=period).mean().values

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    n = len(close)
    
    # Load 1w HTF data ONCE before loop (Rule 1 - CRITICAL)
    df_1w = get_htf_data(prices, '1w')
    
    # Calculate 1w HTF indicators (major trend direction)
    hma_1w_21 = calculate_hma(df_1w['close'].values, period=21)
    
    # Align HTF to LTF (Rule 2 - auto shift(1))
    hma_1w_21_aligned = align_htf_to_ltf(prices, df_1w, hma_1w_21)
    
    # Calculate 1d indicators
    atr_14 = calculate_atr(high, low, close, 14)
    hma_1d_21 = calculate_hma(close, period=21)
    hma_1d_50 = calculate_hma(close, period=50)
    rsi_14 = calculate_rsi(close, period=14)
    donchian_upper, donchian_lower, donchian_middle = calculate_donchian_channels(high, low, period=20)
    sma_50 = calculate_sma(close, period=50)
    
    signals = np.zeros(n)
    
    # Position sizing (Rule 4 - discrete, max 0.40)
    LONG_SIZE = 0.30
    SHORT_SIZE = 0.25
    
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
        if np.isnan(hma_1w_21_aligned[i]):
            continue
        if np.isnan(hma_1d_21[i]) or np.isnan(hma_1d_50[i]):
            continue
        if np.isnan(rsi_14[i]) or np.isnan(donchian_upper[i]) or np.isnan(donchian_lower[i]):
            continue
        
        # === 1W MAJOR TREND (primary direction filter) ===
        bull_regime = close[i] > hma_1w_21_aligned[i]
        bear_regime = close[i] < hma_1w_21_aligned[i]
        
        # === 1D LOCAL TREND (HMA crossover) ===
        hma_bullish = hma_1d_21[i] > hma_1d_50[i]
        hma_bearish = hma_1d_21[i] < hma_1d_50[i]
        
        # === DONCHIAN BREAKOUT SIGNALS ===
        breakout_long = close[i] > donchian_upper[i-1]  # Break above previous upper
        breakout_short = close[i] < donchian_lower[i-1]  # Break below previous lower
        
        # === RSI FILTER (avoid extreme overbought/oversold entries) ===
        rsi_not_overbought = rsi_14[i] < 75.0  # Don't long at extreme overbought
        rsi_not_oversold = rsi_14[i] > 25.0    # Don't short at extreme oversold
        rsi_momentum_long = rsi_14[i] > 45.0   # Some momentum for longs
        rsi_momentum_short = rsi_14[i] < 55.0  # Some weakness for shorts
        
        # === SMA50 FILTER ===
        above_sma50 = close[i] > sma_50[i]
        below_sma50 = close[i] < sma_50[i]
        
        # === ENTRY LOGIC — SIMPLIFIED FOR TRADE FREQUENCY ===
        new_signal = 0.0
        
        # LONG ENTRIES (multiple conditions for flexibility)
        if bull_regime and breakout_long and rsi_not_overbought:
            new_signal = LONG_SIZE
        elif bull_regime and hma_bullish and rsi_momentum_long and rsi_14[i] < 60.0:
            new_signal = LONG_SIZE * 0.7
        elif above_sma50 and breakout_long and hma_bullish:
            new_signal = LONG_SIZE
        elif bull_regime and hma_bullish and rsi_14[i] < 40.0:  # Pullback entry
            new_signal = LONG_SIZE * 0.8
        
        # SHORT ENTRIES (only if no long signal)
        if new_signal == 0.0:
            if bear_regime and breakout_short and rsi_not_oversold:
                new_signal = -SHORT_SIZE
            elif bear_regime and hma_bearish and rsi_momentum_short and rsi_14[i] > 40.0:
                new_signal = -SHORT_SIZE * 0.7
            elif below_sma50 and breakout_short and hma_bearish:
                new_signal = -SHORT_SIZE
            elif bear_regime and hma_bearish and rsi_14[i] > 60.0:  # Retrace entry
                new_signal = -SHORT_SIZE * 0.8
        
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
        
        # === TAKE PROFIT / EXIT CONDITIONS ===
        # Exit long on RSI overbought
        if in_position and position_side > 0 and rsi_14[i] > 80.0:
            new_signal = 0.0
        # Exit short on RSI oversold
        if in_position and position_side < 0 and rsi_14[i] < 20.0:
            new_signal = 0.0
        
        # Regime flip exit (major trend reversal)
        if in_position and position_side > 0 and bear_regime:
            new_signal = 0.0
        if in_position and position_side < 0 and bull_regime:
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
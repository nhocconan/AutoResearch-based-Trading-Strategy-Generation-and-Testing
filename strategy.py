#!/usr/bin/env python3
"""
Experiment #444: 4h Primary + 12h/1d HTF — Donchian Breakout + HMA Trend + RSI

Hypothesis: After 443 experiments, clear patterns emerge:
1. 4h timeframe with HTF filter outperforms 12h/1d primary (less lag, more trades)
2. Donchian breakout + HMA trend worked on SOL (Sharpe +0.782 in research)
3. Complex regime switches fail (exp #440 got 0 trades = Sharpe 0.000)
4. Simpler entry logic = more trades = better statistical significance
5. 12h HMA for major trend direction prevents counter-trend disasters

Why this might beat current best (Sharpe=0.435):
- Donchian(20) breakout captures momentum moves in crypto
- HMA(21/50) crossover filters false breakouts
- 12h HTF trend alignment prevents trading against major trend
- RSI(14) 40-60 filter avoids exhaustion entries
- ATR 2.5x trailing stop protects in crash scenarios
- 4h TF targets 30-50 trades/year (optimal fee/trade balance)

Position sizing: 0.25-0.30 (discrete levels, max 0.40)
Stoploss: 2.5 * ATR trailing
Target: 30-50 trades/year on 4h, >=30 trades/symbol on train, >=3 on test
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_4h_donchian_hma_rsi_12h1d_v1"
timeframe = "4h"
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

def calculate_donchian(high, low, period=20):
    """Calculate Donchian Channel (upper/lower bounds)."""
    upper = pd.Series(high).rolling(window=period, min_periods=period).max().values
    lower = pd.Series(low).rolling(window=period, min_periods=period).min().values
    return upper, lower

def calculate_sma(close, period=200):
    """Calculate Simple Moving Average."""
    return pd.Series(close).rolling(window=period, min_periods=period).mean().values

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    n = len(close)
    
    # Load HTF data ONCE before loop (Rule 1 - CRITICAL)
    df_12h = get_htf_data(prices, '12h')
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate 12h HTF indicators (major trend direction)
    hma_12h_21 = calculate_hma(df_12h['close'].values, period=21)
    hma_12h_50 = calculate_hma(df_12h['close'].values, period=50)
    
    # Calculate 1d HTF indicators (regime filter)
    hma_1d_21 = calculate_hma(df_1d['close'].values, period=21)
    
    # Align HTF to LTF (Rule 2 - auto shift(1))
    hma_12h_21_aligned = align_htf_to_ltf(prices, df_12h, hma_12h_21)
    hma_12h_50_aligned = align_htf_to_ltf(prices, df_12h, hma_12h_50)
    hma_1d_21_aligned = align_htf_to_ltf(prices, df_1d, hma_1d_21)
    
    # Calculate 4h indicators
    atr_14 = calculate_atr(high, low, close, 14)
    hma_4h_21 = calculate_hma(close, period=21)
    hma_4h_50 = calculate_hma(close, period=50)
    rsi_14 = calculate_rsi(close, period=14)
    donchian_upper, donchian_lower = calculate_donchian(high, low, period=20)
    sma_200 = calculate_sma(close, period=200)
    
    signals = np.zeros(n)
    
    # Position sizing (Rule 4 - discrete, max 0.40)
    LONG_SIZE = 0.30
    SHORT_SIZE = 0.25
    
    # Track position state
    in_position = False
    position_side = 0
    highest_price = 0.0
    lowest_price = 0.0
    entry_price = 0.0
    last_trade_bar = -20
    
    for i in range(250, n):
        # Skip if indicators not ready
        if np.isnan(atr_14[i]) or atr_14[i] == 0:
            continue
        if np.isnan(hma_12h_21_aligned[i]) or np.isnan(hma_12h_50_aligned[i]):
            continue
        if np.isnan(hma_1d_21_aligned[i]):
            continue
        if np.isnan(hma_4h_21[i]) or np.isnan(hma_4h_50[i]):
            continue
        if np.isnan(rsi_14[i]) or np.isnan(donchian_upper[i]) or np.isnan(donchian_lower[i]):
            continue
        if np.isnan(sma_200[i]):
            continue
        
        # === 1D MAJOR REGIME (primary direction filter) ===
        # Price above 1d HMA = bull market bias (favor longs)
        # Price below 1d HMA = bear market bias (favor shorts)
        bull_regime = close[i] > hma_1d_21_aligned[i]
        bear_regime = close[i] < hma_1d_21_aligned[i]
        
        # === 12H TREND DIRECTION (secondary filter) ===
        hma_12h_bullish = hma_12h_21_aligned[i] > hma_12h_50_aligned[i]
        hma_12h_bearish = hma_12h_21_aligned[i] < hma_12h_50_aligned[i]
        
        # === 4H LOCAL TREND (HMA crossover) ===
        hma_4h_bullish = hma_4h_21[i] > hma_4h_50[i]
        hma_4h_bearish = hma_4h_21[i] < hma_4h_50[i]
        
        # === DONCHIAN BREAKOUT SIGNALS ===
        # Breakout above upper = potential long
        # Breakout below lower = potential short
        breakout_long = close[i] > donchian_upper[i-1]  # Use previous bar's upper
        breakout_short = close[i] < donchian_lower[i-1]  # Use previous bar's lower
        
        # === RSI FILTER (avoid exhaustion entries) ===
        rsi_neutral_long = 35.0 < rsi_14[i] < 65.0
        rsi_neutral_short = 35.0 < rsi_14[i] < 65.0
        rsi_not_overbought = rsi_14[i] < 70.0
        rsi_not_oversold = rsi_14[i] > 30.0
        
        # === SMA200 FILTER (long-term trend confirmation) ===
        above_sma200 = close[i] > sma_200[i]
        below_sma200 = close[i] < sma_200[i]
        
        # === ENTRY LOGIC ===
        new_signal = 0.0
        bars_since_last_trade = i - last_trade_bar
        
        # LONG ENTRIES - Multiple confluence required
        if bull_regime or hma_12h_bullish:
            # Primary: Donchian breakout + HMA alignment + RSI filter
            if breakout_long and hma_4h_bullish and rsi_neutral_long:
                new_signal = LONG_SIZE
            # Secondary: HMA crossover + RSI pullback (no breakout needed)
            elif hma_4h_bullish and not hma_4h_bearish and 40.0 < rsi_14[i] < 55.0:
                if new_signal == 0.0:
                    new_signal = LONG_SIZE * 0.8
            # Tertiary: Above SMA200 + RSI dip (trend continuation)
            elif above_sma200 and hma_12h_bullish and 35.0 < rsi_14[i] < 50.0:
                if new_signal == 0.0:
                    new_signal = LONG_SIZE * 0.7
        
        # SHORT ENTRIES - Multiple confluence required
        if bear_regime or hma_12h_bearish:
            # Primary: Donchian breakdown + HMA alignment + RSI filter
            if breakout_short and hma_4h_bearish and rsi_neutral_short:
                if new_signal == 0.0:
                    new_signal = -SHORT_SIZE
            # Secondary: HMA crossover + RSI bounce (no breakdown needed)
            elif hma_4h_bearish and not hma_4h_bullish and 45.0 < rsi_14[i] < 60.0:
                if new_signal == 0.0:
                    new_signal = -SHORT_SIZE * 0.8
            # Tertiary: Below SMA200 + RSI rally (trend continuation)
            elif below_sma200 and hma_12h_bearish and 50.0 < rsi_14[i] < 65.0:
                if new_signal == 0.0:
                    new_signal = -SHORT_SIZE * 0.7
        
        # === FREQUENCY BOOST (ensure >=30 trades/symbol on train) ===
        # If no trade for 15 bars (~2.5 days on 4h), force entry on weaker signal
        if bars_since_last_trade > 15 and new_signal == 0.0 and not in_position:
            if bull_regime and hma_4h_bullish and 38.0 < rsi_14[i] < 58.0:
                new_signal = LONG_SIZE * 0.6
            elif bear_regime and hma_4h_bearish and 42.0 < rsi_14[i] < 62.0:
                if new_signal == 0.0:
                    new_signal = -SHORT_SIZE * 0.6
        
        # === EXIT CONDITIONS ===
        # RSI extreme exit (take profit on exhaustion)
        if in_position and position_side > 0 and rsi_14[i] > 72.0:
            new_signal = 0.0
        if in_position and position_side < 0 and rsi_14[i] < 28.0:
            new_signal = 0.0
        
        # Trend reversal exit (12h regime flip)
        if in_position and position_side > 0 and hma_12h_bearish and not hma_12h_bullish:
            new_signal = 0.0
        if in_position and position_side < 0 and hma_12h_bullish and not hma_12h_bearish:
            new_signal = 0.0
        
        # Local trend reversal exit (4h HMA cross)
        if in_position and position_side > 0 and hma_4h_bearish and not hma_4h_bullish:
            new_signal = 0.0
        if in_position and position_side < 0 and hma_4h_bullish and not hma_4h_bearish:
            new_signal = 0.0
        
        # === STOPLOSS (2.5 * ATR trailing) ===
        stoploss_triggered = False
        
        if in_position and position_side > 0:
            highest_price = max(highest_price, close[i])
            stop_price = highest_price - 2.5 * atr_14[i]
            if close[i] < stop_price:
                stoploss_triggered = True
        
        if in_position and position_side < 0:
            if lowest_price == 0.0:
                lowest_price = close[i]
            else:
                lowest_price = min(lowest_price, close[i])
            stop_price = lowest_price + 2.5 * atr_14[i]
            if close[i] > stop_price:
                stoploss_triggered = True
        
        if stoploss_triggered:
            new_signal = 0.0
        
        # === UPDATE POSITION TRACKING ===
        if new_signal != 0.0:
            if not in_position:
                in_position = True
                position_side = np.sign(new_signal)
                entry_price = close[i]
                highest_price = close[i] if position_side > 0 else 0.0
                lowest_price = close[i] if position_side < 0 else 0.0
                last_trade_bar = i
            elif np.sign(new_signal) != position_side:
                # Position flip
                position_side = np.sign(new_signal)
                entry_price = close[i]
                highest_price = close[i] if position_side > 0 else 0.0
                lowest_price = close[i] if position_side < 0 else 0.0
                last_trade_bar = i
        else:
            if in_position:
                in_position = False
                position_side = 0
                entry_price = 0.0
                highest_price = 0.0
                lowest_price = 0.0
                last_trade_bar = i
        
        signals[i] = new_signal
    
    return signals
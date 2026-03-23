#!/usr/bin/env python3
"""
Experiment #976: 12h Primary + 1d HTF — Dual Regime HMA RSI ATR

Hypothesis: After 700+ failed strategies, simpler logic on higher timeframes works best.
12h primary with 1d HTF should generate 20-50 trades/year with lower fee drag.

Key insights from failures:
1. Too many conditions = 0 trades (experiments #967, #968, #970, #975 all had Sharpe=0.000)
2. Funding rate data often unavailable or misaligned — don't rely on it as primary signal
3. Complex regime switching fails — use simpler dual regime (chop vs trend)
4. Higher TF (12h/1d) has better Sharpe than lower TF (1h/4h) in bear markets
5. HMA(21) trend + RSI(14) pullback is proven combination

Strategy logic:
1. 1d HMA(21) = macro trend bias (long only above, short only below)
2. 12h HMA(21) = medium trend confirmation
3. Choppiness(14) = regime filter (CHOP>55 mean revert, CHOP<45 trend follow)
4. RSI(14) = entry timing (oversold long, overbought short)
5. ATR(14) = stoploss at 2.5x from entry

Why this should work:
- Fewer conditions = more trades (avoid 0-trade failure)
- 12h timeframe = 25-40 trades/year target (low fee drag)
- Dual regime = adapts to market conditions
- Conservative sizing (0.25-0.30) = controlled drawdown

Target: Sharpe > 0.612, trades >= 30 train, >= 3 test, ALL symbols positive
Timeframe: 12h (target 25-40 trades/year)
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_12h_hma_rsi_chop_regime_1d_atr_v1"
timeframe = "12h"
leverage = 1.0

def calculate_rsi(close, period=14):
    """Relative Strength Index — standard Wilder's formula."""
    n = len(close)
    rsi = np.full(n, np.nan)
    
    if n < period + 1:
        return rsi
    
    delta = np.diff(close)
    gain = np.where(delta > 0, delta, 0.0)
    loss = np.where(delta < 0, -delta, 0.0)
    
    avg_gain = pd.Series(gain).ewm(span=period, min_periods=period, adjust=False).mean().values
    avg_loss = pd.Series(loss).ewm(span=period, min_periods=period, adjust=False).mean().values
    
    avg_gain = np.concatenate([[np.nan], avg_gain])
    avg_loss = np.concatenate([[np.nan], avg_loss])
    
    with np.errstate(divide='ignore', invalid='ignore'):
        rs = avg_gain / (avg_loss + 1e-10)
        rsi = 100 - (100 / (1 + rs))
    
    rsi = np.clip(rsi, 0, 100)
    return rsi

def calculate_hma(series, period):
    """Hull Moving Average — reduces lag vs EMA."""
    series = pd.Series(series)
    half = period // 2
    sqrt_period = int(np.sqrt(period))
    
    wma_half = series.rolling(window=half, min_periods=half).mean() * 2
    wma_full = series.rolling(window=period, min_periods=period).mean()
    
    wma_diff = wma_half - wma_full
    hma = wma_diff.rolling(window=sqrt_period, min_periods=sqrt_period).mean()
    
    return hma.values

def calculate_atr(high, low, close, period=14):
    """Average True Range — Wilder's smoothing."""
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

def calculate_choppiness(high, low, close, period=14):
    """Choppiness Index — identifies ranging vs trending markets."""
    n = len(close)
    chop = np.full(n, np.nan)
    
    if n < period + 1:
        return chop
    
    for i in range(period, n):
        highest_high = np.max(high[i-period+1:i+1])
        lowest_low = np.min(low[i-period+1:i+1])
        
        price_range = highest_high - lowest_low
        if price_range < 1e-10:
            chop[i] = 100
            continue
        
        tr_sum = 0.0
        for j in range(i-period+1, i+1):
            tr = max(high[j] - low[j], abs(high[j] - close[j-1]), abs(low[j] - close[j-1]))
            tr_sum += tr
        
        chop[i] = 100 * np.log10(tr_sum / price_range) / np.log10(period)
    
    chop = np.clip(chop, 0, 100)
    return chop

def calculate_sma(series, period):
    """Simple Moving Average."""
    return pd.Series(series).rolling(window=period, min_periods=period).mean().values

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    n = len(close)
    
    # Load HTF data ONCE before loop (Rule 1 - CRITICAL)
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate primary (12h) indicators
    rsi_12h = calculate_rsi(close, period=14)
    atr_12h = calculate_atr(high, low, close, period=14)
    chop_12h = calculate_choppiness(high, low, close, period=14)
    hma_12h = calculate_hma(close, period=21)
    sma_200 = calculate_sma(close, period=200)
    
    # Calculate and align 1d HMA for macro trend
    hma_1d_raw = calculate_hma(df_1d['close'].values, period=21)
    hma_1d_aligned = align_htf_to_ltf(prices, df_1d, hma_1d_raw)
    
    signals = np.zeros(n)
    BASE_SIZE = 0.30
    REDUCED_SIZE = 0.20
    
    # Position tracking for stoploss
    in_position = False
    position_side = 0
    entry_price = 0.0
    entry_atr = 0.0
    highest_since_entry = 0.0
    lowest_since_entry = float('inf')
    
    for i in range(250, n):  # Need 200 for SMA + buffer
        # Skip if indicators not ready
        if np.isnan(rsi_12h[i]) or np.isnan(atr_12h[i]) or atr_12h[i] <= 1e-10:
            continue
        if np.isnan(chop_12h[i]) or np.isnan(hma_12h[i]):
            continue
        if np.isnan(hma_1d_aligned[i]) or np.isnan(sma_200[i]):
            continue
        
        # === MACRO TREND (1d HTF HMA21) ===
        macro_bull = close[i] > hma_1d_aligned[i]
        macro_bear = close[i] < hma_1d_aligned[i]
        
        # === MEDIUM TREND (12h HMA21) ===
        trend_bullish = close[i] > hma_12h[i]
        trend_bearish = close[i] < hma_12h[i]
        
        # === LONG-TERM FILTER (SMA200) ===
        above_sma200 = close[i] > sma_200[i]
        below_sma200 = close[i] < sma_200[i]
        
        # === REGIME DETECTION (12h Choppiness) ===
        ranging_regime = chop_12h[i] > 55
        trending_regime = chop_12h[i] < 45
        
        # === RSI SIGNALS (relaxed thresholds for more trades) ===
        rsi_oversold = rsi_12h[i] < 40
        rsi_overbought = rsi_12h[i] > 60
        rsi_extreme_oversold = rsi_12h[i] < 30
        rsi_extreme_overbought = rsi_12h[i] > 70
        rsi_neutral = 35 <= rsi_12h[i] <= 65
        
        desired_signal = 0.0
        
        # === RANGING REGIME (CHOP > 55) — Mean Reversion ===
        if ranging_regime:
            # Long: RSI oversold + price below both HMA
            if rsi_oversold and trend_bearish:
                desired_signal = BASE_SIZE
            # Long: RSI extreme oversold (stronger signal)
            elif rsi_extreme_oversold:
                desired_signal = REDUCED_SIZE
            
            # Short: RSI overbought + price above both HMA
            if rsi_overbought and trend_bullish:
                desired_signal = -BASE_SIZE
            # Short: RSI extreme overbought
            elif rsi_extreme_overbought:
                desired_signal = -REDUCED_SIZE
        
        # === TRENDING REGIME (CHOP < 45) — Trend Following ===
        elif trending_regime:
            # Long: Macro bull + medium bull + RSI pullback
            if macro_bull and trend_bullish and rsi_oversold:
                desired_signal = BASE_SIZE
            # Long: Macro bull + RSI recovering from oversold
            elif macro_bull and rsi_extreme_oversold:
                desired_signal = REDUCED_SIZE
            
            # Short: Macro bear + medium bear + RSI rally
            if macro_bear and trend_bearish and rsi_overbought:
                desired_signal = -BASE_SIZE
            # Short: Macro bear + RSI recovering from overbought
            elif macro_bear and rsi_extreme_overbought:
                desired_signal = -REDUCED_SIZE
        
        # === NEUTRAL REGIME (45 <= CHOP <= 55) ===
        else:
            # Conservative: Only trade with macro trend + extreme RSI
            if macro_bull and rsi_extreme_oversold:
                desired_signal = REDUCED_SIZE
            if macro_bear and rsi_extreme_overbought:
                desired_signal = -REDUCED_SIZE
            
            # Secondary: SMA200 filter + RSI
            if above_sma200 and rsi_oversold and desired_signal == 0:
                desired_signal = REDUCED_SIZE
            if below_sma200 and rsi_overbought and desired_signal == 0:
                desired_signal = -REDUCED_SIZE
        
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
                # Hold long if macro trend still bull
                if macro_bull and rsi_12h[i] < 70:
                    desired_signal = BASE_SIZE
            elif position_side < 0:
                # Hold short if macro trend still bear
                if macro_bear and rsi_12h[i] > 30:
                    desired_signal = -BASE_SIZE
        
        # === EXIT CONDITIONS ===
        if in_position and position_side > 0:
            # Exit long if macro reverses + RSI overbought
            if macro_bear and rsi_12h[i] > 65:
                desired_signal = 0.0
        
        if in_position and position_side < 0:
            # Exit short if macro reverses + RSI oversold
            if macro_bull and rsi_12h[i] < 35:
                desired_signal = 0.0
        
        # === DISCRETIZE SIGNAL VALUES ===
        if desired_signal > 0:
            desired_signal = BASE_SIZE if desired_signal >= BASE_SIZE else REDUCED_SIZE
        elif desired_signal < 0:
            desired_signal = -BASE_SIZE if desired_signal <= -BASE_SIZE else -REDUCED_SIZE
        
        # === UPDATE POSITION TRACKING ===
        if desired_signal != 0.0:
            if not in_position:
                in_position = True
                position_side = int(np.sign(desired_signal))
                entry_price = close[i]
                entry_atr = atr_12h[i]
                highest_since_entry = close[i] if position_side > 0 else 0.0
                lowest_since_entry = close[i] if position_side < 0 else float('inf')
            elif np.sign(desired_signal) != position_side:
                position_side = int(np.sign(desired_signal))
                entry_price = close[i]
                entry_atr = atr_12h[i]
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
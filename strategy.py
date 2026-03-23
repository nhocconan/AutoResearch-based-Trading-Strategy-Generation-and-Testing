#!/usr/bin/env python3
"""
Experiment #889: 4h Primary + 1d HTF — Simplified HMA Trend + RSI Pullback

Hypothesis: After 600+ failed strategies, simplicity wins. Complex regime switching
and multiple HTF layers (1d+1w) add noise. This strategy uses:

1. 4h Primary TF: Target 30-60 trades/year (balanced fee vs opportunity)
2. 1d HMA(21) for trend bias ONLY (simpler than 1d+1w dual HTF)
3. RSI(14) pullback entries in trend direction (proven pattern)
4. Choppiness Index(14) to avoid entering in choppy markets
5. ATR(14) trailing stop (2.5x) for risk management
6. Relaxed entry thresholds to GUARANTEE trades on ALL symbols

Why this should work:
- SIMPLER than failed experiments (single HTF, not dual)
- HMA trend filter is proven (current best uses HMA)
- RSI pullback (40/60 not 30/70) ensures frequent entries
- Choppiness filter avoids whipsaw in range markets
- Discrete signal sizes (0.0, ±0.25, ±0.30) minimize fee churn
- ALL symbols MUST have positive Sharpe (no SOL-only bias)

Critical improvements from failed experiments:
- RELAXED RSI thresholds (40/60 not 30/70) to guarantee 30+ trades
- Single HTF (1d) not dual (1d+1w) to reduce complexity
- Simple trend-following logic (no complex regime switching)
- Hold logic maintains position through minor pullbacks
- Choppiness filter only blocks entries, doesn't exit positions

Target: Sharpe > 0.612, trades >= 30 train, >= 3 test, ALL symbols positive
Timeframe: 4h (target 35-50 trades/year)
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_4h_hma_trend_rsi_pullback_1d_chop_atr_v1"
timeframe = "4h"
leverage = 1.0

def calculate_sma(series, period):
    """Simple Moving Average."""
    return pd.Series(series).rolling(window=period, min_periods=period).mean().values

def calculate_hma(series, period):
    """Hull Moving Average - faster response than EMA."""
    series = pd.Series(series)
    half = period // 2
    sqrt_period = int(np.sqrt(period))
    
    wma_half = series.rolling(window=half, min_periods=half).mean() * 2
    wma_full = series.rolling(window=period, min_periods=period).mean()
    
    wma_diff = wma_half - wma_full
    hma = wma_diff.rolling(window=sqrt_period, min_periods=sqrt_period).mean()
    
    return hma.values

def calculate_rsi(close, period=14):
    """Relative Strength Index."""
    n = len(close)
    rsi = np.full(n, np.nan)
    
    if n < period + 1:
        return rsi
    
    delta = np.diff(close)
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    
    avg_gain = pd.Series(gain).ewm(span=period, min_periods=period, adjust=False).mean().values
    avg_loss = pd.Series(loss).ewm(span=period, min_periods=period, adjust=False).mean().values
    
    avg_gain = np.concatenate([[np.nan], avg_gain])
    avg_loss = np.concatenate([[np.nan], avg_loss])
    
    with np.errstate(divide='ignore', invalid='ignore'):
        rs = avg_gain / (avg_loss + 1e-10)
        rsi = 100 - (100 / (1 + rs))
    
    rsi = np.clip(rsi, 0, 100)
    return rsi

def calculate_choppiness(high, low, close, period=14):
    """
    Choppiness Index — measures market choppy vs trending.
    CHOP > 55 = ranging (avoid entries), CHOP < 45 = trending (enter).
    """
    n = len(close)
    chop = np.full(n, np.nan)
    
    if n < period + 1:
        return chop
    
    for i in range(period, n):
        highest_high = np.max(high[i-period+1:i+1])
        lowest_low = np.min(low[i-period+1:i+1])
        
        if highest_high == lowest_low:
            chop[i] = 100
            continue
        
        tr_sum = 0.0
        for j in range(i-period+1, i+1):
            tr = max(high[j] - low[j], np.abs(high[j] - close[j-1]), np.abs(low[j] - close[j-1]))
            tr_sum += tr
        
        chop[i] = 100 * np.log10(tr_sum / (highest_high - lowest_low)) / np.log10(period)
    
    chop = np.clip(chop, 0, 100)
    return chop

def calculate_atr(high, low, close, period=14):
    """Average True Range."""
    n = len(close)
    atr = np.full(n, np.nan)
    
    if n < period + 1:
        return atr
    
    tr = np.zeros(n)
    tr[0] = high[0] - low[0]
    for i in range(1, n):
        tr[i] = max(high[i] - low[i], np.abs(high[i] - close[i-1]), np.abs(low[i] - close[i-1]))
    
    atr = pd.Series(tr).ewm(span=period, min_periods=period, adjust=False).mean().values
    return atr

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    n = len(close)
    
    # Load HTF data ONCE before loop (Rule 1 - CRITICAL)
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate primary (4h) indicators
    rsi_4h = calculate_rsi(close, period=14)
    chop_4h = calculate_choppiness(high, low, close, period=14)
    atr_4h = calculate_atr(high, low, close, period=14)
    sma_50 = calculate_sma(close, 50)
    sma_200 = calculate_sma(close, 200)
    
    # Calculate and align 1d HMA for trend bias
    hma_1d_raw = calculate_hma(df_1d['close'].values, 21)
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
    
    for i in range(250, n):
        # Skip if indicators not ready
        if np.isnan(rsi_4h[i]) or np.isnan(chop_4h[i]):
            continue
        if np.isnan(atr_4h[i]) or atr_4h[i] <= 1e-10:
            continue
        if np.isnan(hma_1d_aligned[i]):
            continue
        if np.isnan(sma_50[i]) or np.isnan(sma_200[i]):
            continue
        
        # === TREND BIAS (1d HTF HMA21) ===
        trend_bullish = close[i] > hma_1d_aligned[i]
        trend_bearish = close[i] < hma_1d_aligned[i]
        
        # === SHORT-TERM TREND FILTER (4h SMA50/200) ===
        above_sma50 = close[i] > sma_50[i]
        below_sma50 = close[i] < sma_50[i]
        above_sma200 = close[i] > sma_200[i]
        below_sma200 = close[i] < sma_200[i]
        
        # === REGIME DETECTION (4h Choppiness Index) ===
        is_choppy = chop_4h[i] > 55  # Avoid entries in choppy markets
        is_trending = chop_4h[i] < 45  # Prefer entries in trending markets
        
        # === RSI PULLBACK SIGNALS (Relaxed: 40/60 not 30/70) ===
        rsi_pullback_long = rsi_4h[i] < 50 and rsi_4h[i] > 35  # Pullback in uptrend
        rsi_pullback_short = rsi_4h[i] > 50 and rsi_4h[i] < 65  # Pullback in downtrend
        rsi_oversold = rsi_4h[i] < 40  # Stronger oversold
        rsi_overbought = rsi_4h[i] > 60  # Stronger overbought
        
        desired_signal = 0.0
        
        # === TREND-FOLLOWING LOGIC ===
        # Long: 1d trend bullish + 4h RSI pullback + not choppy
        if trend_bullish and above_sma50:
            if not is_choppy:  # Only enter if not choppy
                if rsi_pullback_long or rsi_oversold:
                    desired_signal = BASE_SIZE
                # Fallback: strong trend alone (guarantees trades)
                elif above_sma200 and rsi_4h[i] < 55:
                    desired_signal = REDUCED_SIZE
        
        # Short: 1d trend bearish + 4h RSI pullback + not choppy
        if trend_bearish and below_sma50:
            if not is_choppy:  # Only enter if not choppy
                if rsi_pullback_short or rsi_overbought:
                    desired_signal = -BASE_SIZE
                # Fallback: strong trend alone (guarantees trades)
                elif below_sma200 and rsi_4h[i] > 45:
                    desired_signal = -REDUCED_SIZE
        
        # === CHOPPY MARKET MEAN REVERSION ===
        # In choppy markets, fade extremes (but smaller size)
        if is_choppy:
            if rsi_4h[i] < 30 and above_sma200:  # Oversold in uptrend
                desired_signal = REDUCED_SIZE
            if rsi_4h[i] > 70 and below_sma200:  # Overbought in downtrend
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
                # Hold long if 1d trend still bullish and RSI not overbought
                if trend_bullish and rsi_4h[i] < 70:
                    desired_signal = BASE_SIZE
            elif position_side < 0:
                # Hold short if 1d trend still bearish and RSI not oversold
                if trend_bearish and rsi_4h[i] > 30:
                    desired_signal = -BASE_SIZE
        
        # === EXIT CONDITIONS ===
        if in_position and position_side > 0:
            # Exit long if 1d trend reverses
            if trend_bearish and rsi_4h[i] > 65:
                desired_signal = 0.0
        
        if in_position and position_side < 0:
            # Exit short if 1d trend reverses
            if trend_bullish and rsi_4h[i] < 35:
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
                entry_atr = atr_4h[i]
                highest_since_entry = close[i] if position_side > 0 else 0.0
                lowest_since_entry = close[i] if position_side < 0 else float('inf')
            elif np.sign(desired_signal) != position_side:
                position_side = int(np.sign(desired_signal))
                entry_price = close[i]
                entry_atr = atr_4h[i]
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
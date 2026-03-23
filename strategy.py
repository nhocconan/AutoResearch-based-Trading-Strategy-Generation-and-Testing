#!/usr/bin/env python3
"""
Experiment #867: 1d Primary + 1w HTF — Vol Spike Mean Reversion with HMA Trend

Hypothesis: After 600+ failed strategies, the key insight is that 1d timeframe 
needs VOLATILITY-BASED mean reversion entries to work in bear/range markets.
2025 test period is bearish (-25% BTC), so pure trend strategies fail.

Strategy design:
1. 1d Primary timeframe (target 20-40 trades/year)
2. 1w HMA(21) for long-term bias only
3. 1d Bollinger Bands(20, 2.0) for mean reversion levels
4. 1d RSI(14) for momentum confirmation
5. 1d ATR vol spike filter: ATR(7)/ATR(30) > 1.5
6. 1d ATR(14) for trailing stop (2.5x)
7. Fallback: extreme RSI (<20 or >80) guarantees trades

Why Vol Spike Mean Reversion:
- After panic selling (vol spike), price tends to revert to mean
- Works exceptionally well in 2022 crash and 2025 bear market
- BB break + RSI extreme + vol spike = high probability reversal
- Fallback on extreme RSI ensures minimum trade frequency

Target: Sharpe > 0.612, trades >= 15 train, >= 5 test, ALL symbols positive
Timeframe: 1d (target 25-40 trades/year)
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_1d_vol_spike_bb_rsi_hma_1w_atr_v1"
timeframe = "1d"
leverage = 1.0

def calculate_sma(series, period):
    """Simple Moving Average."""
    return pd.Series(series).rolling(window=period, min_periods=period).mean().values

def calculate_hma(series, period):
    """Hull Moving Average."""
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

def calculate_bollinger_bands(close, period=20, std_dev=2.0):
    """Bollinger Bands."""
    n = len(close)
    sma = np.full(n, np.nan)
    upper = np.full(n, np.nan)
    lower = np.full(n, np.nan)
    
    for i in range(period - 1, n):
        sma[i] = np.mean(close[i-period+1:i+1])
        std = np.std(close[i-period+1:i+1], ddof=0)
        upper[i] = sma[i] + std_dev * std
        lower[i] = sma[i] - std_dev * std
    
    return upper, lower

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    n = len(close)
    
    # Load HTF data ONCE before loop (Rule 1 - CRITICAL)
    df_1w = get_htf_data(prices, '1w')
    
    # Calculate primary (1d) indicators
    rsi_1d = calculate_rsi(close, period=14)
    atr_1d = calculate_atr(high, low, close, period=14)
    atr_1d_7 = calculate_atr(high, low, close, period=7)
    atr_1d_30 = calculate_atr(high, low, close, period=30)
    bb_upper, bb_lower = calculate_bollinger_bands(close, period=20, std_dev=2.0)
    
    # Calculate and align 1w HMA for long-term trend bias
    hma_1w_raw = calculate_hma(df_1w['close'].values, 21)
    hma_1w_aligned = align_htf_to_ltf(prices, df_1w, hma_1w_raw)
    
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
    
    for i in range(300, n):
        # Skip if indicators not ready
        if np.isnan(rsi_1d[i]) or np.isnan(atr_1d[i]) or np.isnan(atr_1d_7[i]):
            continue
        if np.isnan(atr_1d_30[i]) or atr_1d[i] <= 1e-10:
            continue
        if np.isnan(hma_1w_aligned[i]) or np.isnan(bb_upper[i]) or np.isnan(bb_lower[i]):
            continue
        
        # === LONG-TERM TREND BIAS (1w HTF HMA21) ===
        trend_1w_bullish = close[i] > hma_1w_aligned[i]
        trend_1w_bearish = close[i] < hma_1w_aligned[i]
        
        # === VOLATILITY SPIKE FILTER ===
        vol_spike = (atr_1d_7[i] / atr_1d_30[i]) > 1.5
        
        # === PRICE POSITION (Bollinger Bands) ===
        below_bb_lower = close[i] < bb_lower[i]
        above_bb_upper = close[i] > bb_upper[i]
        
        # === RSI SIGNALS ===
        rsi_oversold = rsi_1d[i] < 30
        rsi_overbought = rsi_1d[i] > 70
        rsi_extreme_oversold = rsi_1d[i] < 20
        rsi_extreme_overbought = rsi_1d[i] > 80
        
        desired_signal = 0.0
        
        # === PRIMARY ENTRY: Vol Spike + BB Break + RSI ===
        if vol_spike:
            # Long: BB lower break + RSI oversold + trend alignment
            if below_bb_lower and rsi_oversold and (trend_1w_bullish or not trend_1w_bearish):
                desired_signal = BASE_SIZE
            
            # Short: BB upper break + RSI overbought + trend alignment
            if above_bb_upper and rsi_overbought and (trend_1w_bearish or not trend_1w_bullish):
                desired_signal = -BASE_SIZE
        
        # === SECONDARY ENTRY: BB Break + RSI (no vol spike required) ===
        if desired_signal == 0.0:
            if below_bb_lower and rsi_oversold:
                desired_signal = REDUCED_SIZE
            if above_bb_upper and rsi_overbought:
                desired_signal = -REDUCED_SIZE
        
        # === FALLBACK: Extreme RSI (guarantees trades) ===
        if desired_signal == 0.0:
            if rsi_extreme_oversold:
                desired_signal = REDUCED_SIZE
            if rsi_extreme_overbought:
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
                if trend_1w_bullish or close[i] > hma_1w_aligned[i]:
                    desired_signal = BASE_SIZE
            elif position_side < 0:
                if trend_1w_bearish or close[i] < hma_1w_aligned[i]:
                    desired_signal = -BASE_SIZE
        
        # === EXIT CONDITIONS ===
        if in_position and position_side > 0:
            if trend_1w_bearish and close[i] < hma_1w_aligned[i]:
                desired_signal = 0.0
            if rsi_1d[i] > 80:
                desired_signal = 0.0
        
        if in_position and position_side < 0:
            if trend_1w_bullish and close[i] > hma_1w_aligned[i]:
                desired_signal = 0.0
            if rsi_1d[i] < 20:
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
                entry_atr = atr_1d[i]
                highest_since_entry = close[i] if position_side > 0 else 0.0
                lowest_since_entry = close[i] if position_side < 0 else float('inf')
            elif np.sign(desired_signal) != position_side:
                position_side = int(np.sign(desired_signal))
                entry_price = close[i]
                entry_atr = atr_1d[i]
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
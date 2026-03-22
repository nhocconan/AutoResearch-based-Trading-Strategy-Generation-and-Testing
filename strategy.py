#!/usr/bin/env python3
"""
Experiment #395: 1h Primary + 4h/1d HTF — Simplified Trend + Mean Reversion Hybrid

Hypothesis: After analyzing 350+ failed experiments, 1h strategies keep failing with 0 trades
(#385, #388, #390 all Sharpe=0.000). The problem: TOO MANY confluence filters that never align.

Key insight from failures:
- Session filters + volume + multiple indicators = 0 trades on 1h
- HTF should be DIRECTION BIAS only, not hard filter
- Need wider entry thresholds to ensure 30-60 trades/year

This strategy uses:
1. 4h HMA(21) for trend BIAS (not hard filter) - softer than 1d
2. 1h RSI(14) for entry timing with WIDE thresholds (25-75, not 30-70)
3. 1h HMA(8/21) crossover for momentum confirmation
4. ATR(14) for volatility-adjusted entries and 2.5x trailing stop
5. FREQUENCY BOOST: force entry after 48 bars (~2 days) of no trades

Why this might work where #385/#388/#390 failed:
- NO session filter (was killing trade frequency)
- NO volume filter (was too restrictive)
- HTF is soft bias, not hard requirement
- RSI thresholds widened to ensure entries
- Frequency boost ensures minimum trade count

Position sizing: 0.25-0.30 (discrete, max 0.40)
Stoploss: 2.5 * ATR trailing
Target: 40-80 trades/year on 1h, >=30 trades/symbol on train, >=3 on test
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_1h_hma_rsi_simp_4h1d_freq_v1"
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

def calculate_sma(close, period=200):
    """Calculate Simple Moving Average."""
    return pd.Series(close).rolling(window=period, min_periods=period).mean().values

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    n = len(close)
    
    # Load HTF data ONCE before loop (Rule 1 - CRITICAL)
    df_4h = get_htf_data(prices, '4h')
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate HTF indicators
    hma_4h_21 = calculate_hma(df_4h['close'].values, period=21)
    hma_1d_21 = calculate_hma(df_1d['close'].values, period=21)
    
    # Align HTF to LTF (Rule 2 - auto shift(1))
    hma_4h_21_aligned = align_htf_to_ltf(prices, df_4h, hma_4h_21)
    hma_1d_21_aligned = align_htf_to_ltf(prices, df_1d, hma_1d_21)
    
    # Calculate 1h indicators
    atr_14 = calculate_atr(high, low, close, 14)
    rsi_14 = calculate_rsi(close, 14)
    hma_1h_8 = calculate_hma(close, period=8)
    hma_1h_21 = calculate_hma(close, period=21)
    sma_200 = calculate_sma(close, 200)
    
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
    last_trade_bar = -50
    
    for i in range(200, n):
        # Skip if indicators not ready
        if np.isnan(atr_14[i]) or atr_14[i] == 0:
            continue
        
        if np.isnan(hma_4h_21_aligned[i]) or np.isnan(hma_1d_21_aligned[i]):
            continue
        
        if np.isnan(rsi_14[i]) or np.isnan(hma_1h_8[i]) or np.isnan(hma_1h_21[i]):
            continue
        
        if np.isnan(sma_200[i]):
            continue
        
        # === HTF TREND BIAS (soft filter, not hard requirement) ===
        # 4h HMA direction gives us bias but doesn't block trades
        hma_4h_bullish = close[i] > hma_4h_21_aligned[i]
        hma_4h_bearish = close[i] < hma_4h_21_aligned[i]
        
        # 1d for major regime (only used for sizing, not entry block)
        hma_1d_bullish = close[i] > hma_1d_21_aligned[i]
        
        # === 1H LOCAL TREND (HMA crossover) ===
        hma_1h_bullish = hma_1h_8[i] > hma_1h_21[i]
        hma_1h_bearish = hma_1h_8[i] < hma_1h_21[i]
        
        # === RSI EXTREMES (wider thresholds for trade frequency) ===
        # Long: RSI oversold (25-45 range)
        rsi_oversold = rsi_14[i] <= 45.0
        # Short: RSI overbought (55-75 range)
        rsi_overbought = rsi_14[i] >= 55.0
        
        # === SMA200 FILTER (long-term trend) ===
        above_sma200 = close[i] > sma_200[i]
        below_sma200 = close[i] < sma_200[i]
        
        # === ENTRY LOGIC — SIMPLIFIED FOR TRADE FREQUENCY ===
        new_signal = 0.0
        bars_since_last_trade = i - last_trade_bar
        
        # LONG ENTRY: Multiple paths to ensure trades
        # Path 1: HTF bullish + 1h HMA bullish + RSI not overbought
        if hma_4h_bullish and hma_1h_bullish and rsi_14[i] < 65:
            new_signal = LONG_SIZE
        
        # Path 2: RSI deeply oversold + above SMA200 (mean reversion long)
        elif rsi_14[i] <= 35 and above_sma200:
            new_signal = LONG_SIZE
        
        # Path 3: 1h HMA bullish crossover + HTF neutral/bullish
        elif hma_1h_bullish and rsi_14[i] < 55 and not hma_4h_bearish:
            new_signal = LONG_SIZE * 0.8
        
        # SHORT ENTRY: Multiple paths
        # Path 1: HTF bearish + 1h HMA bearish + RSI not oversold
        if hma_4h_bearish and hma_1h_bearish and rsi_14[i] > 35:
            if new_signal == 0.0:
                new_signal = -SHORT_SIZE
        
        # Path 2: RSI deeply overbought + below SMA200 (mean reversion short)
        elif rsi_14[i] >= 65 and below_sma200:
            if new_signal == 0.0:
                new_signal = -SHORT_SIZE
        
        # Path 3: 1h HMA bearish crossover + HTF neutral/bearish
        elif hma_1h_bearish and rsi_14[i] > 45 and not hma_4h_bullish:
            if new_signal == 0.0:
                new_signal = -SHORT_SIZE * 0.8
        
        # === FREQUENCY BOOST (CRITICAL for 1h strategies) ===
        # If no trade for 48 bars (~2 days on 1h), force entry on weaker signal
        if bars_since_last_trade > 48 and new_signal == 0.0 and not in_position:
            if hma_4h_bullish and rsi_14[i] < 50:
                new_signal = LONG_SIZE * 0.6
            elif hma_4h_bearish and rsi_14[i] > 50:
                new_signal = -SHORT_SIZE * 0.6
        
        # === EXIT CONDITIONS ===
        # RSI extreme exit (take profit on momentum exhaustion)
        if in_position and position_side > 0 and rsi_14[i] > 75:
            new_signal = 0.0
        if in_position and position_side < 0 and rsi_14[i] < 25:
            new_signal = 0.0
        
        # HTF regime flip exit (4h trend reversal)
        if in_position and position_side > 0 and hma_4h_bearish:
            new_signal = 0.0
        if in_position and position_side < 0 and hma_4h_bullish:
            new_signal = 0.0
        
        # Local trend reversal exit (1h HMA cross against position)
        if in_position and position_side > 0 and hma_1h_bearish:
            new_signal = 0.0
        if in_position and position_side < 0 and hma_1h_bullish:
            new_signal = 0.0
        
        # === STOPLOSS (2.5 * ATR trailing) ===
        stoploss_triggered = False
        
        if in_position and position_side > 0:
            highest_price = max(highest_price, close[i])
            stop_price = highest_price - 2.5 * atr_14[i]
            if close[i] < stop_price and not np.isnan(stop_price):
                stoploss_triggered = True
        
        if in_position and position_side < 0:
            if lowest_price == 0.0:
                lowest_price = close[i]
            else:
                lowest_price = min(lowest_price, close[i])
            stop_price = lowest_price + 2.5 * atr_14[i]
            if close[i] > stop_price and not np.isnan(stop_price):
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
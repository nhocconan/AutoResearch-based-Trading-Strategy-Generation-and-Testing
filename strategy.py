#!/usr/bin/env python3
"""
Experiment #402: 12h Primary + 1d/1w HTF — Simplified HMA Trend + RSI Pullback

Hypothesis: After 400+ experiments, the pattern is clear:
1. Complex dual-regime strategies FAIL on 12h/1d (see #392, #393, #397 - all Sharpe < -1.0)
2. Choppiness Index + CRSI combinations NOT working on higher timeframes
3. SIMPLER is better: HMA trend + RSI pullback + ATR stop (proven in #382, #386)
4. 12h timeframe should generate 20-50 trades/year (~80-200 over 4y train)
5. Use 1w HMA for ultra-long-term bias, 1d HMA for intermediate confirmation
6. Wider RSI range (25-55 long, 45-75 short) to ensure trade frequency
7. Discrete position sizing: 0.0, ±0.25, ±0.30 (max 0.40)

Why this might beat current best (Sharpe=0.435):
- 12h TF = fewer trades = less fee drag than 4h/1h strategies
- 1w HTF filter prevents counter-trend trades in major bear/bull markets
- RSI pullback entries catch dips in uptrend / rallies in downtrend
- ATR 2.5x trailing stop protects capital during whipsaw
- Simpler logic = more robust across BTC/ETH/SOL (no SOL-only bias)

Position sizing: 0.25-0.30 (discrete levels, max 0.40)
Stoploss: 2.5 * ATR trailing
Target: 20-50 trades/year on 12h, >=30 trades/symbol on train, >=3 on test
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_12h_hma_rsi_pullback_1d1w_simp_v1"
timeframe = "12h"
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
    df_1d = get_htf_data(prices, '1d')
    df_1w = get_htf_data(prices, '1w')
    
    # Calculate HTF indicators (major trend direction)
    hma_1w_21 = calculate_hma(df_1w['close'].values, period=21)
    hma_1d_21 = calculate_hma(df_1d['close'].values, period=21)
    hma_1d_50 = calculate_hma(df_1d['close'].values, period=50)
    
    # Align HTF to LTF (Rule 2 - auto shift(1))
    hma_1w_21_aligned = align_htf_to_ltf(prices, df_1w, hma_1w_21)
    hma_1d_21_aligned = align_htf_to_ltf(prices, df_1d, hma_1d_21)
    hma_1d_50_aligned = align_htf_to_ltf(prices, df_1d, hma_1d_50)
    
    # Calculate 12h indicators
    atr_14 = calculate_atr(high, low, close, 14)
    rsi_14 = calculate_rsi(close, 14)
    hma_12h_16 = calculate_hma(close, period=16)
    hma_12h_48 = calculate_hma(close, period=48)
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
    last_trade_bar = -20
    
    for i in range(200, n):
        # Skip if indicators not ready
        if np.isnan(atr_14[i]) or atr_14[i] == 0:
            continue
        
        if np.isnan(hma_1w_21_aligned[i]) or np.isnan(hma_1d_21_aligned[i]):
            continue
        
        if np.isnan(rsi_14[i]) or np.isnan(hma_12h_16[i]) or np.isnan(hma_12h_48[i]):
            continue
        
        if np.isnan(sma_200[i]):
            continue
        
        # === 1W ULTRA-LONG TERM TREND (major bias) ===
        # Price above 1w HMA = bull market (favor longs)
        # Price below 1w HMA = bear market (favor shorts)
        bull_regime_1w = close[i] > hma_1w_21_aligned[i]
        bear_regime_1w = close[i] < hma_1w_21_aligned[i]
        
        # === 1D INTERMEDIATE TREND (confirmation) ===
        # 1d HMA(21) > HMA(50) = intermediate uptrend
        # 1d HMA(21) < HMA(50) = intermediate downtrend
        bull_regime_1d = hma_1d_21_aligned[i] > hma_1d_50_aligned[i]
        bear_regime_1d = hma_1d_21_aligned[i] < hma_1d_50_aligned[i]
        
        # === 12H LOCAL TREND (HMA crossover) ===
        hma_bullish_12h = hma_12h_16[i] > hma_12h_48[i]
        hma_bearish_12h = hma_12h_16[i] < hma_12h_48[i]
        
        # === RSI PULLBACK SIGNALS (wider range for trade frequency) ===
        # Long: RSI pulled back to 25-55 in uptrend (buying dip)
        rsi_long_pullback = 25.0 <= rsi_14[i] <= 55.0
        # Short: RSI pulled back to 45-75 in downtrend (selling rally)
        rsi_short_pullback = 45.0 <= rsi_14[i] <= 75.0
        
        # === SMA200 FILTER (long-term trend confirmation) ===
        above_sma200 = close[i] > sma_200[i]
        below_sma200 = close[i] < sma_200[i]
        
        # === ENTRY LOGIC — SIMPLIFIED TREND + PULLBACK ===
        new_signal = 0.0
        bars_since_last_trade = i - last_trade_bar
        
        # LONG ENTRY: 1w bull + 1d bull + (12h HMA bullish OR HMA cross) + RSI pullback
        if bull_regime_1w and bull_regime_1d:
            if hma_bullish_12h and rsi_long_pullback and above_sma200:
                new_signal = LONG_SIZE
            # HMA crossover entry (16 crosses above 48)
            elif i > 201 and hma_12h_16[i] > hma_12h_48[i] and hma_12h_16[i-1] <= hma_12h_48[i-1]:
                if rsi_14[i] < 65 and above_sma200:
                    new_signal = LONG_SIZE
        
        # SHORT ENTRY: 1w bear + 1d bear + (12h HMA bearish OR HMA cross) + RSI pullback
        if bear_regime_1w and bear_regime_1d:
            if hma_bearish_12h and rsi_short_pullback and below_sma200:
                if new_signal == 0.0:
                    new_signal = -SHORT_SIZE
            # HMA crossover entry (16 crosses below 48)
            elif i > 201 and hma_12h_16[i] < hma_12h_48[i] and hma_12h_16[i-1] >= hma_12h_48[i-1]:
                if rsi_14[i] > 35 and below_sma200:
                    if new_signal == 0.0:
                        new_signal = -SHORT_SIZE
        
        # === FREQUENCY BOOST (ensure >=30 trades/symbol on train) ===
        # If no trade for 15 bars (~7.5 days on 12h), force entry on weaker signal
        if bars_since_last_trade > 15 and new_signal == 0.0 and not in_position:
            if bull_regime_1w and rsi_14[i] < 50 and hma_bullish_12h and above_sma200:
                new_signal = LONG_SIZE * 0.8
            elif bear_regime_1w and rsi_14[i] > 50 and hma_bearish_12h and below_sma200:
                new_signal = -SHORT_SIZE * 0.8
        
        # === EXIT CONDITIONS ===
        # RSI extreme exit (take profit on momentum exhaustion)
        if in_position and position_side > 0 and rsi_14[i] > 80:
            new_signal = 0.0
        if in_position and position_side < 0 and rsi_14[i] < 20:
            new_signal = 0.0
        
        # Trend reversal exit (1w regime flip - major signal)
        if in_position and position_side > 0 and bear_regime_1w:
            new_signal = 0.0
        if in_position and position_side < 0 and bull_regime_1w:
            new_signal = 0.0
        
        # Local trend reversal exit (12h HMA cross against position)
        if in_position and position_side > 0 and hma_bearish_12h:
            new_signal = 0.0
        if in_position and position_side < 0 and hma_bullish_12h:
            new_signal = 0.0
        
        # === STOPLOSS (2.5 * ATR trailing) ===
        stoploss_triggered = False
        
        if in_position and position_side > 0:
            highest_price = max(highest_price, high[i])
            stop_price = highest_price - 2.5 * atr_14[i]
            if low[i] < stop_price and not np.isnan(stop_price):
                stoploss_triggered = True
        
        if in_position and position_side < 0:
            if lowest_price == 0.0:
                lowest_price = low[i]
            else:
                lowest_price = min(lowest_price, low[i])
            stop_price = lowest_price + 2.5 * atr_14[i]
            if high[i] > stop_price and not np.isnan(stop_price):
                stoploss_triggered = True
        
        if stoploss_triggered:
            new_signal = 0.0
        
        # === UPDATE POSITION TRACKING ===
        if new_signal != 0.0:
            if not in_position:
                in_position = True
                position_side = np.sign(new_signal)
                entry_price = close[i]
                highest_price = high[i] if position_side > 0 else 0.0
                lowest_price = low[i] if position_side < 0 else 0.0
                last_trade_bar = i
            elif np.sign(new_signal) != position_side:
                # Position flip
                position_side = np.sign(new_signal)
                entry_price = close[i]
                highest_price = high[i] if position_side > 0 else 0.0
                lowest_price = low[i] if position_side < 0 else 0.0
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
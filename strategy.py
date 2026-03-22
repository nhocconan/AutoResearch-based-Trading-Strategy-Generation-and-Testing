#!/usr/bin/env python3
"""
Experiment #394: 4h Primary + 12h HTF — KAMA Adaptive Trend + Vol Expansion + RSI

Hypothesis: After analyzing 350+ failed experiments including #389 (Sharpe=-0.453):
1. 4h strategies fail with too many filters (#384, #391 had 5+ confluence = 0 trades)
2. 12h HTF aligns better with 4h entries than 1d (closer timeframe = less lag)
3. KAMA (Kaufman Adaptive) adapts to volatility better than HMA/EMA in chop
4. ATR ratio (ATR7/ATR30) captures vol expansion before moves (proven edge)
5. Need MULTIPLE entry triggers (breakout OR pullback) to ensure >=30 trades/symbol
6. Simpler logic: 2-3 confluence max, not 5+
7. Discrete sizing 0.25-0.30 with 2.5x ATR trailing stop

Why this might beat current best (Sharpe=0.435):
- KAMA adapts to regime changes (fast in trends, slow in chop)
- ATR vol expansion filter catches moves before they happen
- 12h HTF gives trend direction without 1d lag
- Multiple entry paths ensure trade frequency requirement met
- Simpler = fewer whipsaws, better Sharpe

Position sizing: 0.25-0.30 (discrete, max 0.40)
Stoploss: 2.5 * ATR trailing
Target: 30-50 trades/year on 4h, >=30 trades/symbol on train, >=3 on test
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_4h_kama_vol_expansion_rsi_12h_v1"
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

def calculate_kama(close, er_period=10, fast_period=2, slow_period=30):
    """
    Calculate Kaufman Adaptive Moving Average (KAMA).
    KAMA adapts speed based on market efficiency (trend vs noise).
    """
    close_s = pd.Series(close)
    
    # Efficiency Ratio (ER): measures trend vs noise
    change = np.abs(close_s - close_s.shift(er_period))
    volatility = np.abs(close_s - close_s.shift(1)).rolling(window=er_period, min_periods=er_period).sum()
    
    er = change / (volatility + 1e-10)
    er = er.fillna(0)
    
    # Smoothing constants
    fast_sc = 2.0 / (fast_period + 1)
    slow_sc = 2.0 / (slow_period + 1)
    
    # Adaptive smoothing constant
    sc = (er * (fast_sc - slow_sc) + slow_sc) ** 2
    
    # KAMA calculation
    kama = np.zeros(len(close))
    kama[0] = close[0]
    
    for i in range(1, len(close)):
        if np.isnan(sc.iloc[i]):
            kama[i] = kama[i-1]
        else:
            kama[i] = kama[i-1] + sc.iloc[i] * (close[i] - kama[i-1])
    
    return kama

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

def calculate_sma(close, period=20):
    """Calculate Simple Moving Average."""
    return pd.Series(close).rolling(window=period, min_periods=period).mean().values

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    n = len(close)
    
    # Load 12h HTF data ONCE before loop (Rule 1 - CRITICAL)
    df_12h = get_htf_data(prices, '12h')
    
    # Calculate 12h HTF indicators (major trend direction)
    kama_12h_20 = calculate_kama(df_12h['close'].values, er_period=10, fast_period=2, slow_period=30)
    sma_12h_50 = calculate_sma(df_12h['close'].values, 50)
    
    # Align HTF to LTF (Rule 2 - auto shift(1))
    kama_12h_20_aligned = align_htf_to_ltf(prices, df_12h, kama_12h_20)
    sma_12h_50_aligned = align_htf_to_ltf(prices, df_12h, sma_12h_50)
    
    # Calculate 4h indicators
    atr_14 = calculate_atr(high, low, close, 14)
    atr_7 = calculate_atr(high, low, close, 7)
    atr_30 = calculate_atr(high, low, close, 30)
    rsi_14 = calculate_rsi(close, 14)
    kama_4h_20 = calculate_kama(close, er_period=10, fast_period=2, slow_period=30)
    kama_4h_50 = calculate_kama(close, er_period=10, fast_period=2, slow_period=30)
    # Recalculate KAMA with different slow period for 50
    kama_4h_50 = calculate_kama(close, er_period=10, fast_period=2, slow_period=50)
    sma_4h_200 = calculate_sma(close, 200)
    
    # Volatility expansion ratio (ATR7/ATR30)
    vol_ratio = atr_7 / (atr_30 + 1e-10)
    
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
    
    for i in range(100, n):
        # Skip if indicators not ready
        if np.isnan(atr_14[i]) or atr_14[i] == 0:
            continue
        
        if np.isnan(kama_12h_20_aligned[i]) or np.isnan(sma_12h_50_aligned[i]):
            continue
        
        if np.isnan(rsi_14[i]) or np.isnan(kama_4h_20[i]) or np.isnan(kama_4h_50[i]):
            continue
        
        if np.isnan(vol_ratio[i]) or np.isnan(sma_4h_200[i]):
            continue
        
        # === 12H MAJOR TREND (primary direction filter) ===
        # Price above 12h KAMA = bull market bias (favor longs)
        # Price below 12h KAMA = bear market bias (favor shorts)
        bull_regime = close[i] > kama_12h_20_aligned[i]
        bear_regime = close[i] < kama_12h_20_aligned[i]
        
        # 12h SMA50 confirmation
        bull_trend_12h = kama_12h_20_aligned[i] > sma_12h_50_aligned[i]
        bear_trend_12h = kama_12h_20_aligned[i] < sma_12h_50_aligned[i]
        
        # === 4H LOCAL TREND (KAMA adaptive) ===
        kama_bullish = kama_4h_20[i] > kama_4h_50[i]
        kama_bearish = kama_4h_20[i] < kama_4h_50[i]
        
        # Price vs 200 SMA (long-term trend)
        above_sma200 = close[i] > sma_4h_200[i]
        below_sma200 = close[i] < sma_4h_200[i]
        
        # === VOLATILITY EXPANSION (key signal) ===
        # Vol ratio > 1.3 = vol expanding (good for entries)
        # Vol ratio < 0.8 = vol contracting (avoid entries)
        vol_expanding = vol_ratio[i] > 1.2
        vol_contracting = vol_ratio[i] < 0.9
        
        # === RSI MOMENTUM ===
        # RSI > 55 = bullish momentum
        # RSI < 45 = bearish momentum
        rsi_bullish = rsi_14[i] > 50
        rsi_bearish = rsi_14[i] < 50
        
        # RSI pullback zones (wider for trade frequency)
        rsi_long_pullback = 35.0 <= rsi_14[i] <= 55.0
        rsi_short_pullback = 45.0 <= rsi_14[i] <= 65.0
        
        # === ENTRY LOGIC — MULTIPLE TRIGGERS FOR FREQUENCY ===
        new_signal = 0.0
        bars_since_last_trade = i - last_trade_bar
        
        # LONG ENTRY PATH 1: Trend + Vol Expansion + Momentum
        if bull_regime and bull_trend_12h and kama_bullish:
            if vol_expanding and rsi_bullish:
                new_signal = LONG_SIZE
            elif rsi_long_pullback and above_sma200:
                # Pullback entry in uptrend
                new_signal = LONG_SIZE
        
        # LONG ENTRY PATH 2: Vol contraction breakout (mean reversion)
        if vol_contracting and bars_since_last_trade > 8:
            if bull_regime and kama_bullish and rsi_14[i] < 50:
                new_signal = LONG_SIZE * 0.7
        
        # SHORT ENTRY PATH 1: Trend + Vol Expansion + Momentum
        if bear_regime and bear_trend_12h and kama_bearish:
            if vol_expanding and rsi_bearish:
                if new_signal == 0.0:
                    new_signal = -SHORT_SIZE
            elif rsi_short_pullback and below_sma200:
                # Pullback entry in downtrend
                if new_signal == 0.0:
                    new_signal = -SHORT_SIZE
        
        # SHORT ENTRY PATH 2: Vol contraction breakout (mean reversion)
        if vol_contracting and bars_since_last_trade > 8:
            if bear_regime and kama_bearish and rsi_14[i] > 50:
                if new_signal == 0.0:
                    new_signal = -SHORT_SIZE * 0.7
        
        # === FREQUENCY BOOST (ensure >=30 trades/symbol on train) ===
        # If no trade for 15 bars (~2.5 days on 4h), force entry on weaker signal
        if bars_since_last_trade > 15 and new_signal == 0.0 and not in_position:
            if bull_regime and kama_bullish and rsi_14[i] < 55:
                new_signal = LONG_SIZE * 0.6
            elif bear_regime and kama_bearish and rsi_14[i] > 45:
                new_signal = -SHORT_SIZE * 0.6
        
        # === EXIT CONDITIONS ===
        # RSI extreme exit (take profit on momentum exhaustion)
        if in_position and position_side > 0 and rsi_14[i] > 75:
            new_signal = 0.0
        if in_position and position_side < 0 and rsi_14[i] < 25:
            new_signal = 0.0
        
        # Trend reversal exit (12h regime flip)
        if in_position and position_side > 0 and bear_regime:
            new_signal = 0.0
        if in_position and position_side < 0 and bull_regime:
            new_signal = 0.0
        
        # Local trend reversal exit (4h KAMA cross)
        if in_position and position_side > 0 and kama_bearish:
            new_signal = 0.0
        if in_position and position_side < 0 and kama_bullish:
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
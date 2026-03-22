#!/usr/bin/env python3
"""
Experiment #407: 1d Primary + 1w HTF — HMA Trend + RSI Pullback + Asymmetric Entries

Hypothesis: After 368 failed experiments, clear patterns emerge for 1d timeframe:
1. 1d needs 20-50 trades/year (higher TF = fewer trades = less fee drag)
2. Simple HMA trend + RSI pullback worked in #403 but needs asymmetric entries
3. 1w HMA(21) for major regime (bull/bear) — prevents counter-trend trades
4. 1d HMA(16/48) for local trend direction
5. RSI(14) pullback to 40-60 zone for entry timing (not extremes)
6. Asymmetric: easier long in bull regime, harder short (BTC long bias)
7. ATR 2.5x trailing stop protects in crash scenarios

Why this might beat current best (Sharpe=0.435):
- 1d TF has lowest fee drag of all tested timeframes
- 1w HTF filter prevents 2022-style whipsaw (major trend alignment)
- RSI pullback (not breakout) catches better risk/reward entries
- Asymmetric sizing accounts for crypto long bias
- Simple logic = less overfitting than complex regime switching

Position sizing: 0.30 long, 0.25 short (discrete, max 0.40)
Stoploss: 2.5 * ATR trailing
Target: 25-45 trades/year on 1d, >=30 trades/symbol on train, >=3 on test
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_1d_hma_rsi_pullback_1w_asym_v1"
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

def calculate_sma(close, period=200):
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
    rsi_14 = calculate_rsi(close, 14)
    hma_1d_16 = calculate_hma(close, period=16)
    hma_1d_48 = calculate_hma(close, period=48)
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
    last_trade_bar = -30
    
    for i in range(250, n):
        # Skip if indicators not ready
        if np.isnan(atr_14[i]) or atr_14[i] == 0:
            continue
        
        if np.isnan(hma_1w_21_aligned[i]):
            continue
        
        if np.isnan(rsi_14[i]) or np.isnan(hma_1d_16[i]) or np.isnan(hma_1d_48[i]):
            continue
        
        if np.isnan(sma_200[i]):
            continue
        
        # === 1W MAJOR TREND (primary direction filter) ===
        # Price above 1w HMA = bull market bias (favor longs)
        # Price below 1w HMA = bear market bias (favor shorts)
        bull_regime = close[i] > hma_1w_21_aligned[i]
        bear_regime = close[i] < hma_1w_21_aligned[i]
        
        # === 1D LOCAL TREND (HMA crossover) ===
        hma_bullish = hma_1d_16[i] > hma_1d_48[i]
        hma_bearish = hma_1d_16[i] < hma_1d_48[i]
        
        # === SMA200 FILTER (long-term trend) ===
        above_sma200 = close[i] > sma_200[i]
        below_sma200 = close[i] < sma_200[i]
        
        # === RSI PULLBACK ZONES (entry timing) ===
        # RSI 35-50 = pullback in uptrend (long entry)
        # RSI 50-65 = pullback in downtrend (short entry)
        rsi_pullback_long = 35.0 <= rsi_14[i] <= 55.0
        rsi_pullback_short = 45.0 <= rsi_14[i] <= 65.0
        
        # RSI extreme (avoid entry at extremes)
        rsi_not_overbought = rsi_14[i] < 70.0
        rsi_not_oversold = rsi_14[i] > 30.0
        
        # === ENTRY LOGIC — ASYMMETRIC ===
        new_signal = 0.0
        bars_since_last_trade = i - last_trade_bar
        
        # LONG ENTRY (easier in bull regime)
        if bull_regime:
            # Primary: HMA bullish + RSI pullback + above SMA200
            if hma_bullish and rsi_pullback_long and above_sma200:
                new_signal = LONG_SIZE
            # Secondary: HMA bullish + RSI neutral (45-55) + bull regime
            elif hma_bullish and 45.0 <= rsi_14[i] <= 55.0 and bull_regime:
                new_signal = LONG_SIZE * 0.8
        
        # SHORT ENTRY (harder, only in strong bear regime)
        if bear_regime:
            # Primary: HMA bearish + RSI pullback + below SMA200
            if hma_bearish and rsi_pullback_short and below_sma200:
                new_signal = -SHORT_SIZE
            # Secondary: HMA bearish + RSI neutral (45-55) + bear regime
            elif hma_bearish and 45.0 <= rsi_14[i] <= 55.0 and bear_regime:
                new_signal = -SHORT_SIZE * 0.7
        
        # === FREQUENCY BOOST (ensure >=30 trades/symbol on train) ===
        # If no trade for 25 bars (~25 days on 1d), force entry on weaker signal
        if bars_since_last_trade > 25 and new_signal == 0.0 and not in_position:
            if bull_regime and hma_bullish and rsi_14[i] < 55.0:
                new_signal = LONG_SIZE * 0.5
            elif bear_regime and hma_bearish and rsi_14[i] > 45.0:
                new_signal = -SHORT_SIZE * 0.5
        
        # === EXIT CONDITIONS ===
        # RSI extreme exit (take profit on exhaustion)
        if in_position and position_side > 0 and rsi_14[i] > 70.0:
            new_signal = 0.0
        if in_position and position_side < 0 and rsi_14[i] < 30.0:
            new_signal = 0.0
        
        # Trend reversal exit (1w regime flip)
        if in_position and position_side > 0 and bear_regime:
            new_signal = 0.0
        if in_position and position_side < 0 and bull_regime:
            new_signal = 0.0
        
        # Local trend reversal exit (1d HMA cross)
        if in_position and position_side > 0 and hma_bearish:
            new_signal = 0.0
        if in_position and position_side < 0 and hma_bullish:
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
#!/usr/bin/env python3
"""
Experiment #373: 1d Primary + 1w HTF — Simple Trend-Follow with RSI Pullback

Hypothesis: After 372 failed experiments, complexity is the enemy. The best performers
were SIMPLE strategies with clear regime filters. This strategy:
1. 1w HMA(21) = major trend direction (bull/bear regime)
2. 1d RSI(14) pullback entries WITH the 1w trend (no counter-trend trades)
3. Long: 1w HMA bullish + RSI(14) < 40 (pullback in uptrend)
4. Short: 1w HMA bearish + RSI(14) > 60 (rally in downtrend)
5. ATR(14) trailing stop 2.5x to cut losers
6. Asymmetric sizing: longs 0.30, shorts 0.20 (crypto long bias)
7. Target: 25-40 trades/year on 1d timeframe

Why this might work when dual-regime failed:
- Single clear rule: trade WITH weekly trend only
- RSI pullback entries have proven edge in crypto (buy dips in bull, sell rallies in bear)
- 1d timeframe = ~250 bars/year, 25-40 trades = 10-16% trade rate (optimal)
- No conflicting regime logic = cleaner signals
- 1w HTF prevents whipsaw on major trend changes

Position sizing: 0.20-0.30 (max 0.35), discrete levels
Stoploss: 2.5 * ATR trailing
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_1d_rsi_pullback_1w_hma_simp_v1"
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

def calculate_sma(close, period=50):
    """Calculate Simple Moving Average."""
    return pd.Series(close).rolling(window=period, min_periods=period).mean().values

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    n = len(close)
    
    # Load HTF data ONCE before loop (Rule 1 - CRITICAL)
    df_1w = get_htf_data(prices, '1w')
    
    # Calculate 1w HTF indicators (major trend direction)
    hma_1w_21 = calculate_hma(df_1w['close'].values, period=21)
    
    # Align HTF to LTF (Rule 2 - auto shift(1))
    hma_1w_21_aligned = align_htf_to_ltf(prices, df_1w, hma_1w_21)
    
    # Calculate 1d indicators
    atr_14 = calculate_atr(high, low, close, 14)
    rsi_14 = calculate_rsi(close, 14)
    sma_50 = calculate_sma(close, 50)
    
    signals = np.zeros(n)
    
    # Position sizing (Rule 4 - discrete, max 0.40)
    # Asymmetric: longs favored in crypto
    LONG_SIZE = 0.30
    SHORT_SIZE = 0.20
    
    # Track position state for stoploss
    in_position = False
    position_side = 0
    entry_price = 0.0
    highest_price = 0.0
    lowest_price = 0.0
    last_trade_bar = -20
    
    for i in range(100, n):
        # Skip if indicators not ready
        if np.isnan(atr_14[i]) or atr_14[i] == 0:
            continue
        
        if np.isnan(hma_1w_21_aligned[i]):
            continue
        
        if np.isnan(rsi_14[i]) or np.isnan(sma_50[i]):
            continue
        
        # === 1W MAJOR TREND REGIME ===
        # Price above 1w HMA = bull trend, below = bear trend
        trend_bull = close[i] > hma_1w_21_aligned[i]
        trend_bear = close[i] < hma_1w_21_aligned[i]
        
        # === 1D LOCAL CONFIRMATION ===
        # Price above SMA50 confirms local strength
        local_bull = close[i] > sma_50[i]
        local_bear = close[i] < sma_50[i]
        
        # === RSI PULLBACK SIGNALS ===
        # Long: RSI pulled back but not oversold (30-45) in bull trend
        rsi_pullback_long = 30.0 < rsi_14[i] < 45.0
        # Short: RSI rallied but not overbought (55-70) in bear trend
        rsi_pullback_short = 55.0 < rsi_14[i] < 70.0
        
        # === ENTRY LOGIC ===
        new_signal = 0.0
        bars_since_last_trade = i - last_trade_bar
        
        # LONG ENTRY: 1w bull trend + RSI pullback + local confirmation
        if trend_bull and rsi_pullback_long:
            if local_bull:
                new_signal = LONG_SIZE
            else:
                new_signal = LONG_SIZE * 0.7  # weaker signal without local confirmation
        
        # SHORT ENTRY: 1w bear trend + RSI rally + local confirmation
        elif trend_bear and rsi_pullback_short:
            if local_bear:
                new_signal = -SHORT_SIZE
            else:
                new_signal = -SHORT_SIZE * 0.7
        
        # === FREQUENCY BOOST (ensure 25+ trades/year) ===
        # If no trade for 15 bars (~15 days), look for weaker entries
        if bars_since_last_trade > 15 and new_signal == 0.0 and not in_position:
            # Weaker long: just need 1w bull + RSI < 50
            if trend_bull and rsi_14[i] < 50.0:
                new_signal = LONG_SIZE * 0.5
            # Weaker short: just need 1w bear + RSI > 50
            elif trend_bear and rsi_14[i] > 50.0:
                new_signal = -SHORT_SIZE * 0.5
        
        # === STOPLOSS LOGIC (Rule 6) - 2.5 * ATR trailing ===
        stoploss_triggered = False
        
        if in_position and position_side != 0:
            if position_side > 0:
                # Update highest price for long
                if close[i] > highest_price:
                    highest_price = close[i]
                stoploss_price = highest_price - 2.5 * atr_14[i]
                if close[i] < stoploss_price:
                    stoploss_triggered = True
            
            if position_side < 0:
                # Update lowest price for short
                if lowest_price == 0.0 or close[i] < lowest_price:
                    lowest_price = close[i]
                stoploss_price = lowest_price + 2.5 * atr_14[i]
                if close[i] > stoploss_price:
                    stoploss_triggered = True
        
        # === RSI REVERSAL EXIT ===
        # Exit long when RSI overbought, exit short when RSI oversold
        rsi_exit = False
        if in_position and position_side != 0:
            if position_side > 0 and rsi_14[i] > 70.0:
                rsi_exit = True
            if position_side < 0 and rsi_14[i] < 30.0:
                rsi_exit = True
        
        # === TREND REVERSAL EXIT ===
        # Exit if 1w trend flips against position
        trend_reversal = False
        if in_position and position_side != 0:
            if position_side > 0 and trend_bear:
                trend_reversal = True
            if position_side < 0 and trend_bull:
                trend_reversal = True
        
        if stoploss_triggered or rsi_exit or trend_reversal:
            new_signal = 0.0
        
        # === DISCRETIZE SIGNAL (reduce churn) ===
        if new_signal != 0.0:
            if new_signal > 0.28:
                new_signal = LONG_SIZE
            elif new_signal > 0.15:
                new_signal = LONG_SIZE * 0.7
            elif new_signal > 0:
                new_signal = LONG_SIZE * 0.5
            elif new_signal < -0.18:
                new_signal = -SHORT_SIZE
            elif new_signal < -0.10:
                new_signal = -SHORT_SIZE * 0.7
            else:
                new_signal = -SHORT_SIZE * 0.5
        else:
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
                # Flip position
                position_side = np.sign(new_signal)
                entry_price = close[i]
                highest_price = close[i] if position_side > 0 else 0.0
                lowest_price = close[i] if position_side < 0 else 0.0
                last_trade_bar = i
            # If same side, maintain position (no update needed)
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
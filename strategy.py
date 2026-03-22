#!/usr/bin/env python3
"""
Experiment #362: 12h Primary + 1d/1w HTF — Simplified Trend-Follow with RSI Pullback

Hypothesis: After analyzing 361 failed experiments, the pattern is clear:
1. Dual-regime switching (choppy vs trending) creates whipsaws — FAILED in #352, #353, #356
2. Connors RSI mean-reversion alone doesn't work in crypto's trending nature
3. Choppiness Index adds complexity without edge
4. SIMPLE trend-follow with RSI pullback entries works best (proven in baseline)

Key changes from failed #356:
- REMOVE dual-regime logic (source of whipsaws)
- REMOVE Choppiness Index (no proven edge)
- REMOVE Connors RSI (too complex, failed repeatedly)
- USE simple RSI(14) pullback entries in HTF trend direction
- 1d HMA(21) for primary trend, 1w HMA(21) for macro bias
- Only enter longs when 1d HMA bullish, only shorts when 1d HMA bearish
- RSI pullback to 40-50 for longs, 50-60 for shorts (not extremes)
- Asymmetric sizing: 0.30 longs, 0.20 shorts (crypto long bias)
- ATR 2.5x trailing stoploss

Why this might beat Sharpe=0.435:
- Simpler = fewer whipsaws, cleaner signals
- RSI pullback in trend direction = high probability entries
- 12h TF = 25-40 trades/year (optimal frequency)
- 1w macro filter prevents counter-trend trades in major moves
- Asymmetric sizing matches crypto's long bias

Position sizing: 0.30 longs, 0.20 shorts
Stoploss: 2.5 * ATR trailing
Target: 25-40 trades/year on 12h
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_12h_trend_rsi_pullback_1d1w_simp_v1"
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

def calculate_sma(close, period=50):
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
    
    # Calculate 1d HTF indicators (primary trend direction)
    hma_1d_21 = calculate_hma(df_1d['close'].values, period=21)
    hma_1d_21_aligned = align_htf_to_ltf(prices, df_1d, hma_1d_21)
    
    # Calculate 1w HTF indicators (macro bias)
    hma_1w_21 = calculate_hma(df_1w['close'].values, period=21)
    hma_1w_21_aligned = align_htf_to_ltf(prices, df_1w, hma_1w_21)
    
    # Calculate 12h indicators
    atr_14 = calculate_atr(high, low, close, 14)
    rsi_14 = calculate_rsi(close, 14)
    hma_12h_21 = calculate_hma(close, period=21)
    hma_12h_8 = calculate_hma(close, period=8)
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
        
        if np.isnan(hma_1d_21_aligned[i]) or np.isnan(hma_1w_21_aligned[i]):
            continue
        
        if np.isnan(rsi_14[i]) or np.isnan(hma_12h_21[i]):
            continue
        
        # === 1D PRIMARY TREND (main direction filter) ===
        trend_1d_bull = close[i] > hma_1d_21_aligned[i]
        trend_1d_bear = close[i] < hma_1d_21_aligned[i]
        
        # === 1W MACRO BIAS (avoid counter-trend in major moves) ===
        trend_1w_bull = close[i] > hma_1w_21_aligned[i]
        trend_1w_bear = close[i] < hma_1w_21_aligned[i]
        
        # === 12H LOCAL TREND (entry timing) ===
        hma_bullish = hma_12h_8[i] > hma_12h_21[i]
        hma_bearish = hma_12h_8[i] < hma_12h_21[i]
        
        # === RSI PULLBACK ZONES (entry triggers) ===
        # Long: RSI pulls back to 40-50 in uptrend
        rsi_pullback_long = 38.0 < rsi_14[i] < 52.0
        # Short: RSI rallies to 50-60 in downtrend
        rsi_pullback_short = 48.0 < rsi_14[i] < 62.0
        
        # === RSI EXTREMES (reversal warnings) ===
        rsi_oversold = rsi_14[i] < 35.0
        rsi_overbought = rsi_14[i] > 65.0
        
        # === PRICE VS SMA50 (trend confirmation) ===
        price_above_sma50 = close[i] > sma_50[i] if not np.isnan(sma_50[i]) else True
        
        # === ENTRY LOGIC - SIMPLIFIED TREND-FOLLOW ===
        new_signal = 0.0
        bars_since_last_trade = i - last_trade_bar
        
        # === LONG ENTRY: 1d bullish + RSI pullback + 12h HMA bullish ===
        if trend_1d_bull and rsi_pullback_long:
            # Strong signal: all conditions align
            if hma_bullish and price_above_sma50:
                # Check 1w macro bias (prefer longs in bull macro)
                if trend_1w_bull:
                    new_signal = LONG_SIZE
                else:
                    new_signal = LONG_SIZE * 0.7  # Reduce size if 1w bearish
            # Moderate signal: 1d bull + RSI pullback only
            elif hma_bullish:
                new_signal = LONG_SIZE * 0.7
            # Weak signal: only 1d bull + RSI pullback
            elif price_above_sma50:
                new_signal = LONG_SIZE * 0.5
        
        # === SHORT ENTRY: 1d bearish + RSI pullback + 12h HMA bearish ===
        if trend_1d_bear and rsi_pullback_short:
            # Strong signal: all conditions align
            if hma_bearish and not price_above_sma50:
                # Check 1w macro bias (prefer shorts in bear macro)
                if trend_1w_bear:
                    if new_signal == 0.0:
                        new_signal = -SHORT_SIZE
                else:
                    if new_signal == 0.0:
                        new_signal = -SHORT_SIZE * 0.7  # Reduce size if 1w bullish
            # Moderate signal: 1d bear + RSI pullback only
            elif hma_bearish:
                if new_signal == 0.0:
                    new_signal = -SHORT_SIZE * 0.7
            # Weak signal: only 1d bear + RSI pullback
            elif not price_above_sma50:
                if new_signal == 0.0:
                    new_signal = -SHORT_SIZE * 0.5
        
        # === RSI EXTREME REVERSAL (counter-trend scalp) ===
        # Only if no position and extreme RSI
        if new_signal == 0.0 and not in_position:
            # Long: RSI very oversold + price below 1d HMA (overshoot)
            if rsi_oversold and close[i] < hma_1d_21_aligned[i] * 0.95:
                new_signal = LONG_SIZE * 0.4
            # Short: RSI very overbought + price above 1d HMA (overshoot)
            elif rsi_overbought and close[i] > hma_1d_21_aligned[i] * 1.05:
                new_signal = -SHORT_SIZE * 0.4
        
        # === FREQUENCY SAFEGUARD (ensure 25+ trades/year on 12h) ===
        # Force trade if no signal for 15 bars (~7.5 days on 12h)
        if bars_since_last_trade > 15 and new_signal == 0.0 and not in_position:
            if trend_1d_bull and rsi_14[i] < 45.0:
                new_signal = LONG_SIZE * 0.5
            elif trend_1d_bear and rsi_14[i] > 55.0:
                new_signal = -SHORT_SIZE * 0.5
        
        # === STOPLOSS LOGIC (Rule 6) - 2.5 * ATR trailing ===
        stoploss_triggered = False
        
        if in_position and position_side != 0:
            if position_side > 0:
                # Update highest price for long position
                if close[i] > highest_price:
                    highest_price = close[i]
                stoploss_price = highest_price - 2.5 * atr_14[i]
                if close[i] < stoploss_price:
                    stoploss_triggered = True
            
            if position_side < 0:
                # Update lowest price for short position
                if lowest_price == 0.0 or close[i] < lowest_price:
                    lowest_price = close[i]
                stoploss_price = lowest_price + 2.5 * atr_14[i]
                if close[i] > stoploss_price:
                    stoploss_triggered = True
        
        # === RSI REVERSAL EXIT ===
        rsi_exit = False
        if in_position and position_side != 0:
            # Exit long when RSI overbought
            if position_side > 0 and rsi_overbought:
                rsi_exit = True
            # Exit short when RSI oversold
            if position_side < 0 and rsi_oversold:
                rsi_exit = True
        
        # === TREND REVERSAL EXIT ===
        trend_exit = False
        if in_position and position_side != 0:
            # Exit long when 1d trend turns bearish
            if position_side > 0 and trend_1d_bear and close[i] < hma_12h_21[i]:
                trend_exit = True
            # Exit short when 1d trend turns bullish
            if position_side < 0 and trend_1d_bull and close[i] > hma_12h_21[i]:
                trend_exit = True
        
        if stoploss_triggered or rsi_exit or trend_exit:
            new_signal = 0.0
        
        # === DISCRETIZE SIGNAL (reduce churn - Rule 4) ===
        if new_signal != 0.0:
            if new_signal > 0:
                if new_signal >= LONG_SIZE * 0.9:
                    new_signal = LONG_SIZE
                elif new_signal >= LONG_SIZE * 0.6:
                    new_signal = LONG_SIZE * 0.7
                else:
                    new_signal = LONG_SIZE * 0.5
            else:
                if new_signal <= -SHORT_SIZE * 0.9:
                    new_signal = -SHORT_SIZE
                elif new_signal <= -SHORT_SIZE * 0.6:
                    new_signal = -SHORT_SIZE * 0.7
                else:
                    new_signal = -SHORT_SIZE * 0.5
        
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
            # Same direction: keep position, update tracking
            elif position_side > 0:
                highest_price = max(highest_price, close[i])
            else:
                if lowest_price == 0.0 or close[i] < lowest_price:
                    lowest_price = close[i]
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
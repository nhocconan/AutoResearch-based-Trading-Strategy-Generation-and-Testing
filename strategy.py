#!/usr/bin/env python3
"""
Experiment #364: 4h Primary + 12h/1d HTF — Simplified Trend-Pullback Strategy

Hypothesis: After 363 experiments, the clearest pattern is:
1. Complex dual-regime strategies overfit and fail (exp #352-363 all negative Sharpe)
2. Simple trend-following with pullback entries works best (current best Sharpe=0.435)
3. 4h timeframe balances trade frequency (25-50/year) vs fee drag
4. 12h HMA(21) for trend direction + 1d HMA(21) for major trend filter
5. RSI(14) pullback entries: 35-50 for longs, 50-65 for shorts (relaxed from prior)
6. ATR(14) 2.5x trailing stop to cut losers quickly
7. Discrete position sizing: 0.25-0.30 to minimize fee churn
8. ROC(10) momentum confirmation to avoid dead-cat bounces

Why this might beat current best (Sharpe=0.435):
- Simpler logic = less overfitting across BTC/ETH/SOL
- Relaxed RSI thresholds = more trades (critical for all symbols)
- 4h TF = optimal trade frequency without 15m/30m fee drag
- Momentum confirmation filters out weak signals

Position sizing: 0.25-0.30 (discrete levels)
Stoploss: 2.5 * ATR trailing
Target: 30-50 trades/year on 4h (must hit on ALL symbols)
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_4h_hma_rsi_roc_12h1d_simp_v1"
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

def calculate_roc(close, period=10):
    """Calculate Rate of Change (momentum)."""
    close_s = pd.Series(close)
    roc = close_s.pct_change(periods=period) * 100.0
    return roc.values

def calculate_sma(close, period=50):
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
    
    # Calculate 12h HTF indicators (trend direction)
    hma_12h_21 = calculate_hma(df_12h['close'].values, period=21)
    
    # Calculate 1d HTF indicators (major trend filter)
    hma_1d_21 = calculate_hma(df_1d['close'].values, period=21)
    
    # Align HTF to LTF (Rule 2 - auto shift(1) for completed bars)
    hma_12h_21_aligned = align_htf_to_ltf(prices, df_12h, hma_12h_21)
    hma_1d_21_aligned = align_htf_to_ltf(prices, df_1d, hma_1d_21)
    
    # Calculate 4h indicators
    atr_14 = calculate_atr(high, low, close, 14)
    rsi_14 = calculate_rsi(close, 14)
    roc_10 = calculate_roc(close, 10)
    sma_50 = calculate_sma(close, 50)
    hma_4h_21 = calculate_hma(close, period=21)
    hma_4h_8 = calculate_hma(close, period=8)
    
    signals = np.zeros(n)
    
    # Position sizing (Rule 4 - discrete, max 0.40)
    LONG_BASE = 0.25
    LONG_STRONG = 0.30
    SHORT_BASE = 0.20
    SHORT_STRONG = 0.25
    
    # Track position state
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
        
        if np.isnan(hma_12h_21_aligned[i]) or np.isnan(hma_1d_21_aligned[i]):
            continue
        
        if np.isnan(rsi_14[i]) or np.isnan(roc_10[i]):
            continue
        
        if np.isnan(hma_4h_21[i]) or np.isnan(sma_50[i]):
            continue
        
        # === 1D MAJOR TREND (primary filter) ===
        # Price above 1d HMA = bull market bias (favor longs)
        # Price below 1d HMA = bear market bias (favor shorts)
        major_trend_bull = close[i] > hma_1d_21_aligned[i]
        major_trend_bear = close[i] < hma_1d_21_aligned[i]
        
        # === 12H TREND DIRECTION ===
        trend_bull = close[i] > hma_12h_21_aligned[i]
        trend_bear = close[i] < hma_12h_21_aligned[i]
        
        # === 4H LOCAL TREND ===
        hma_bullish = hma_4h_8[i] > hma_4h_21[i]
        hma_bearish = hma_4h_8[i] < hma_4h_21[i]
        
        # === MOMENTUM CONFIRMATION ===
        momentum_positive = roc_10[i] > 0.5
        momentum_negative = roc_10[i] < -0.5
        
        # === RSI PULLBACK LEVELS (relaxed for more trades) ===
        # Long entry: RSI 35-50 (pullback in uptrend)
        # Short entry: RSI 50-65 (pullback in downtrend)
        rsi_pullback_long = 35.0 <= rsi_14[i] <= 52.0
        rsi_pullback_short = 48.0 <= rsi_14[i] <= 65.0
        rsi_oversold = rsi_14[i] < 35.0
        rsi_overbought = rsi_14[i] > 65.0
        
        # === PRICE POSITION ===
        price_above_sma50 = close[i] > sma_50[i]
        price_below_sma50 = close[i] < sma_50[i]
        
        # === ENTRY LOGIC ===
        new_signal = 0.0
        bars_since_last_trade = i - last_trade_bar
        
        # === LONG ENTRY CONDITIONS ===
        # Must have: major trend bull OR 12h trend bull
        # Plus: RSI pullback + momentum confirmation
        long_confidence = 0
        
        if major_trend_bull or trend_bull:
            if rsi_pullback_long and momentum_positive:
                long_confidence = 2
            elif rsi_pullback_long or rsi_oversold:
                long_confidence = 1
            elif rsi_14[i] < 45.0 and price_above_sma50:
                long_confidence = 1
        
        if long_confidence >= 2:
            new_signal = LONG_STRONG
        elif long_confidence == 1:
            new_signal = LONG_BASE
        
        # === SHORT ENTRY CONDITIONS ===
        # Must have: major trend bear OR 12h trend bear
        # Plus: RSI pullback + momentum confirmation
        short_confidence = 0
        
        if major_trend_bear or trend_bear:
            if rsi_pullback_short and momentum_negative:
                short_confidence = 2
            elif rsi_pullback_short or rsi_overbought:
                short_confidence = 1
            elif rsi_14[i] > 55.0 and price_below_sma50:
                short_confidence = 1
        
        if short_confidence >= 2 and new_signal == 0.0:
            new_signal = -SHORT_STRONG
        elif short_confidence >= 1 and new_signal == 0.0:
            new_signal = -SHORT_BASE
        
        # === FREQUENCY BOOSTER (ensure 30+ trades/year) ===
        # If no trade for 10 bars (~40 hours on 4h), force entry on weaker signals
        if bars_since_last_trade > 10 and new_signal == 0.0 and not in_position:
            if trend_bull and rsi_14[i] < 45.0:
                new_signal = LONG_BASE * 0.7
            elif trend_bear and rsi_14[i] > 55.0:
                new_signal = -SHORT_BASE * 0.7
            elif major_trend_bull and rsi_14[i] < 50.0:
                new_signal = LONG_BASE * 0.5
            elif major_trend_bear and rsi_14[i] > 50.0:
                new_signal = -SHORT_BASE * 0.5
        
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
        rsi_exit = False
        if in_position and position_side != 0:
            if position_side > 0 and rsi_overbought:
                rsi_exit = True
            if position_side < 0 and rsi_oversold:
                rsi_exit = True
        
        # === TREND REVERSAL EXIT ===
        trend_exit = False
        if in_position and position_side != 0:
            if position_side > 0 and trend_bear and hma_bearish:
                trend_exit = True
            if position_side < 0 and trend_bull and hma_bullish:
                trend_exit = True
        
        if stoploss_triggered or rsi_exit or trend_exit:
            new_signal = 0.0
        
        # === DISCRETIZE SIGNAL (reduce churn) ===
        if new_signal != 0.0:
            if new_signal > 0.28:
                new_signal = LONG_STRONG
            elif new_signal > 0.15:
                new_signal = LONG_BASE
            elif new_signal < -0.23:
                new_signal = -SHORT_STRONG
            elif new_signal < -0.12:
                new_signal = -SHORT_BASE
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
            # If same direction, keep position (no update needed)
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
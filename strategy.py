#!/usr/bin/env python3
"""
Experiment #379: 4h Primary + 1d HTF — Simplified Regime-Adaptive Strategy

Hypothesis: After 344+ failed experiments, the pattern is clear:
1. Over-complicated dual-regime logic creates 0-trade strategies (exp #375, #378)
2. Simple trend + pullback works best (current best Sharpe=0.435 uses HMA+RSI)
3. 4h timeframe with 1d HTF should generate 25-45 trades/year
4. Choppiness Index for regime detection (proven on ETH Sharpe +0.923)
5. KAMA for adaptive trend (better than EMA in choppy markets)
6. RSI for pullback entries (not extremes — use 40/60 not 30/70 for more trades)
7. Asymmetric sizing: longs 0.25-0.30, shorts 0.20-0.25 (crypto long bias)
8. ATR trailing stop 2.5x to cut losers

Why this might beat current best (Sharpe=0.435):
- KAMA adapts to volatility better than HMA in crypto
- 4h TF generates more signals than 12h/1d while avoiding 15m/30m fee drag
- Simpler entry logic ensures 30+ trades (avoiding 0-trade failure)
- 1d KAMA for major trend, 4h for entries (proven MTF pattern)

Position sizing: 0.25-0.30 longs, 0.20-0.25 shorts
Stoploss: 2.5 * ATR trailing
Target: 30-50 trades/year on 4h
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_4h_kama_regime_rsi_1d_v1"
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
    KAMA adapts to market noise - fast in trends, slow in chop.
    ER = |close - close[n]| / sum(|close[i] - close[i-1]|)
    SC = (ER * (fast_sc - slow_sc) + slow_sc)^2
    KAMA = KAMA_prev + SC * (close - KAMA_prev)
    """
    n = len(close)
    kama = np.zeros(n)
    
    # Efficiency Ratio
    er = np.zeros(n)
    for i in range(er_period, n):
        signal = np.abs(close[i] - close[i - er_period])
        noise = np.sum(np.abs(np.diff(close[i - er_period:i + 1])))
        if noise > 0:
            er[i] = signal / noise
        else:
            er[i] = 0
    
    # Smoothing Constant
    fast_sc = 2.0 / (fast_period + 1)
    slow_sc = 2.0 / (slow_period + 1)
    sc = (er * (fast_sc - slow_sc) + slow_sc) ** 2
    
    # KAMA calculation
    kama[er_period] = close[er_period]
    for i in range(er_period + 1, n):
        kama[i] = kama[i - 1] + sc[i] * (close[i] - kama[i - 1])
    
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

def calculate_choppiness(high, low, close, period=14):
    """
    Calculate Choppiness Index (CHOP).
    CHOP = 100 * LOG10(SUM(ATR, n) / (Highest High - Lowest Low)) / LOG10(n)
    CHOP > 61.8 = choppy/ranging market (mean-revert)
    CHOP < 38.2 = trending market (trend-follow)
    """
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]
    atr_vals = pd.Series(tr).ewm(span=period, min_periods=period, adjust=False).mean().values
    
    atr_sum = pd.Series(atr_vals).rolling(window=period, min_periods=period).sum().values
    highest_high = pd.Series(high).rolling(window=period, min_periods=period).max().values
    lowest_low = pd.Series(low).rolling(window=period, min_periods=period).min().values
    
    price_range = highest_high - lowest_low
    
    with np.errstate(divide='ignore', invalid='ignore'):
        chop = 100.0 * np.log10(atr_sum / (price_range + 1e-10)) / np.log10(period)
    
    chop = np.clip(chop, 0, 100)
    chop[np.isnan(chop)] = 50.0
    
    return chop

def calculate_adx(high, low, close, period=14):
    """Calculate ADX (Average Directional Index)."""
    n = len(close)
    
    plus_dm = np.zeros(n)
    minus_dm = np.zeros(n)
    
    for i in range(1, n):
        plus_move = high[i] - high[i - 1]
        minus_move = low[i - 1] - low[i]
        
        if plus_move > minus_move and plus_move > 0:
            plus_dm[i] = plus_move
        if minus_move > plus_move and minus_move > 0:
            minus_dm[i] = minus_move
    
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]
    
    atr = pd.Series(tr).ewm(span=period, min_periods=period, adjust=False).mean().values
    
    plus_di = 100.0 * pd.Series(plus_dm).ewm(span=period, min_periods=period, adjust=False).mean().values / (atr + 1e-10)
    minus_di = 100.0 * pd.Series(minus_dm).ewm(span=period, min_periods=period, adjust=False).mean().values / (atr + 1e-10)
    
    dx = 100.0 * np.abs(plus_di - minus_di) / (plus_di + minus_di + 1e-10)
    adx = pd.Series(dx).ewm(span=period, min_periods=period, adjust=False).mean().values
    
    return adx, plus_di, minus_di

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    n = len(close)
    
    # Load HTF data ONCE before loop (Rule 1 - CRITICAL)
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate 1d HTF indicators (major trend direction)
    kama_1d_21 = calculate_kama(df_1d['close'].values, er_period=10, fast_period=2, slow_period=30)
    
    # Align HTF to LTF (Rule 2 - auto shift(1))
    kama_1d_21_aligned = align_htf_to_ltf(prices, df_1d, kama_1d_21)
    
    # Calculate 4h indicators
    atr_14 = calculate_atr(high, low, close, 14)
    chop_14 = calculate_choppiness(high, low, close, 14)
    rsi_14 = calculate_rsi(close, 14)
    kama_4h_21 = calculate_kama(close, er_period=10, fast_period=2, slow_period=30)
    kama_4h_10 = calculate_kama(close, er_period=5, fast_period=2, slow_period=20)
    adx_14, plus_di, minus_di = calculate_adx(high, low, close, 14)
    
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
        
        if np.isnan(kama_1d_21_aligned[i]):
            continue
        
        if np.isnan(chop_14[i]) or np.isnan(rsi_14[i]):
            continue
        
        if np.isnan(kama_4h_21[i]) or np.isnan(adx_14[i]):
            continue
        
        # === 1D MAJOR TREND REGIME (primary direction filter) ===
        regime_bull = close[i] > kama_1d_21_aligned[i]
        regime_bear = close[i] < kama_1d_21_aligned[i]
        
        # === CHOPPINESS REGIME (determines strategy type) ===
        choppy_regime = chop_14[i] > 55.0  # Slightly lower threshold for more trades
        trending_regime = chop_14[i] < 45.0  # Slightly higher threshold for more trades
        
        # === TREND STRENGTH (ADX) ===
        strong_trend = adx_14[i] > 25.0
        weak_trend = adx_14[i] < 20.0
        
        # === 4H LOCAL TREND ===
        kama_bullish = kama_4h_10[i] > kama_4h_21[i]
        kama_bearish = kama_4h_10[i] < kama_4h_21[i]
        
        # === RSI SIGNALS (adjusted for more trades) ===
        rsi_oversold = rsi_14[i] < 40.0  # More lenient than 30
        rsi_overbought = rsi_14[i] > 60.0  # More lenient than 70
        rsi_neutral = 40.0 <= rsi_14[i] <= 60.0
        
        # === DI CROSSOVER ===
        di_bullish = plus_di[i] > minus_di[i]
        di_bearish = plus_di[i] < minus_di[i]
        
        # === ENTRY LOGIC - SIMPLIFIED DUAL REGIME ===
        new_signal = 0.0
        bars_since_last_trade = i - last_trade_bar
        
        # === TRENDING REGIME: TREND-FOLLOW ===
        if trending_regime and strong_trend:
            # Long: bull regime + KAMA bullish + RSI pullback (not oversold) + DI bullish
            if regime_bull and kama_bullish and di_bullish:
                if rsi_14[i] < 55.0:  # Pullback entry
                    new_signal = LONG_STRONG
                elif close[i] > kama_4h_10[i]:  # Momentum continuation
                    new_signal = LONG_BASE
            
            # Short: bear regime + KAMA bearish + RSI rally (not overbought) + DI bearish
            if regime_bear and kama_bearish and di_bearish:
                if new_signal == 0.0:
                    if rsi_14[i] > 45.0:  # Rally entry
                        new_signal = -SHORT_STRONG
                    elif close[i] < kama_4h_10[i]:
                        new_signal = -SHORT_BASE
        
        # === CHOPPY REGIME: MEAN-REVERSION ===
        elif choppy_regime:
            # Long: RSI oversold + bull regime preference
            if rsi_oversold:
                if regime_bull or kama_bullish:
                    new_signal = LONG_BASE
                else:
                    new_signal = LONG_BASE * 0.6
            
            # Short: RSI overbought + bear regime preference
            if rsi_overbought and new_signal == 0.0:
                if regime_bear or kama_bearish:
                    new_signal = -SHORT_BASE
                else:
                    new_signal = -SHORT_BASE * 0.6
        
        # === NEUTRAL REGIME: HYBRID ===
        else:
            # Use trend bias with RSI timing
            if regime_bull and kama_bullish and rsi_14[i] < 50.0:
                new_signal = LONG_BASE
            elif regime_bear and kama_bearish and rsi_14[i] > 50.0:
                new_signal = -SHORT_BASE
            elif rsi_oversold and regime_bull:
                new_signal = LONG_BASE * 0.7
            elif rsi_overbought and regime_bear:
                new_signal = -SHORT_BASE * 0.7
        
        # === FREQUENCY BOOST (ensure 30+ trades/year on 4h) ===
        # Force trade if no signal for 15 bars (~2.5 days on 4h)
        if bars_since_last_trade > 15 and new_signal == 0.0 and not in_position:
            if regime_bull and rsi_14[i] < 50.0:
                new_signal = LONG_BASE * 0.5
            elif regime_bear and rsi_14[i] > 50.0:
                new_signal = -SHORT_BASE * 0.5
            elif kama_bullish and rsi_14[i] < 45.0:
                new_signal = LONG_BASE * 0.5
            elif kama_bearish and rsi_14[i] > 55.0:
                new_signal = -SHORT_BASE * 0.5
        
        # === STOPLOSS LOGIC (Rule 6) - 2.5 * ATR trailing ===
        stoploss_triggered = False
        
        if in_position and position_side != 0:
            if position_side > 0:
                if close[i] > highest_price:
                    highest_price = close[i]
                stoploss_price = highest_price - 2.5 * atr_14[i]
                if close[i] < stoploss_price:
                    stoploss_triggered = True
            
            if position_side < 0:
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
        
        # === REGIME REVERSAL EXIT ===
        regime_reversal = False
        if in_position and position_side != 0:
            if position_side > 0 and regime_bear and close[i] < kama_4h_21[i]:
                regime_reversal = True
            if position_side < 0 and regime_bull and close[i] > kama_4h_21[i]:
                regime_reversal = True
        
        if stoploss_triggered or rsi_exit or regime_reversal:
            new_signal = 0.0
        
        # === DISCRETIZE SIGNAL (reduce churn) ===
        if new_signal != 0.0:
            if abs(new_signal) < 0.10:
                new_signal = 0.0
            elif new_signal > 0.27:
                new_signal = LONG_STRONG
            elif new_signal > 0:
                new_signal = LONG_BASE
            elif new_signal < -0.22:
                new_signal = -SHORT_STRONG
            else:
                new_signal = -SHORT_BASE
        
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
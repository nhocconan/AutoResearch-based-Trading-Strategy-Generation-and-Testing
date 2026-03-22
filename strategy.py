#!/usr/bin/env python3
"""
Experiment #361: 4h Primary + 1d HTF — Simplified Trend-Follow with RSI Pullback

Hypothesis: After 360 experiments, the clearest pattern is:
1. Complex dual-regime strategies OVERFIT and fail (exp #352-356 all negative Sharpe)
2. SIMPLER trend-follow with pullback entries works best (current best: Sharpe=0.435)
3. 4h timeframe generates 20-50 trades/year — optimal frequency for fee/capture balance
4. 1d HMA(21) for major trend bias (not 1w which is too slow for 4h entries)
5. 4h RSI(14) pullback entries in trend direction (buy dips in uptrend, sell rallies in downtrend)
6. ATR(14) trailing stop at 2.5x to cut losers quickly
7. Asymmetric sizing: longs 0.25-0.30, shorts 0.15-0.20 (crypto long bias)

Why this might beat current best (Sharpe=0.435):
- Simpler logic = less overfitting, more robust across BTC/ETH/SOL
- 4h TF captures trends better than 12h/1d while avoiding 15m/30m fee drag
- RSI pullback entries improve entry timing vs pure breakout
- 1d HTF provides strong trend filter without lag of 1w

Position sizing: 0.25-0.30 longs, 0.15-0.20 shorts
Stoploss: 2.5 * ATR trailing
Target: 25-45 trades/year on 4h
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_4h_hma_rsi_pullback_1d_simp_v3"
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

def calculate_adx(high, low, close, period=14):
    """Calculate ADX (Average Directional Index)."""
    high_s = pd.Series(high)
    low_s = pd.Series(low)
    close_s = pd.Series(close)
    
    plus_dm = high_s.diff()
    minus_dm = -low_s.diff()
    
    plus_dm = plus_dm.where((plus_dm > minus_dm) & (plus_dm > 0), 0.0)
    minus_dm = minus_dm.where((minus_dm > plus_dm) & (minus_dm > 0), 0.0)
    
    tr1 = high_s - low_s
    tr2 = (high_s - close_s.shift(1)).abs()
    tr3 = (low_s - close_s.shift(1)).abs()
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    
    atr = tr.ewm(span=period, min_periods=period, adjust=False).mean()
    
    plus_di = 100.0 * plus_dm.ewm(span=period, min_periods=period, adjust=False).mean() / atr
    minus_di = 100.0 * minus_dm.ewm(span=period, min_periods=period, adjust=False).mean() / atr
    
    dx = 100.0 * (plus_di - minus_di).abs() / (plus_di + minus_di + 1e-10)
    adx = dx.ewm(span=period, min_periods=period, adjust=False).mean()
    
    return adx.values

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    n = len(close)
    
    # Load HTF data ONCE before loop (Rule 1 - CRITICAL)
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate 1d HTF indicators (major trend direction)
    hma_1d_21 = calculate_hma(df_1d['close'].values, period=21)
    
    # Align HTF to LTF (Rule 2 - auto shift(1))
    hma_1d_21_aligned = align_htf_to_ltf(prices, df_1d, hma_1d_21)
    
    # Calculate 4h indicators
    atr_14 = calculate_atr(high, low, close, 14)
    rsi_14 = calculate_rsi(close, 14)
    hma_4h_21 = calculate_hma(close, period=21)
    hma_4h_8 = calculate_hma(close, period=8)
    adx_14 = calculate_adx(high, low, close, 14)
    
    signals = np.zeros(n)
    
    # Position sizing (Rule 4 - discrete, max 0.40)
    # Asymmetric: longs favored in crypto
    LONG_BASE = 0.25
    LONG_STRONG = 0.30
    SHORT_BASE = 0.15
    SHORT_STRONG = 0.20
    
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
        
        if np.isnan(hma_1d_21_aligned[i]):
            continue
        
        if np.isnan(rsi_14[i]) or np.isnan(hma_4h_21[i]):
            continue
        
        # === 1D MAJOR TREND REGIME (primary direction filter) ===
        # Price above 1d HMA = bull bias (favor longs)
        # Price below 1d HMA = bear bias (favor shorts)
        regime_bull = close[i] > hma_1d_21_aligned[i]
        regime_bear = close[i] < hma_1d_21_aligned[i]
        
        # === 4H LOCAL TREND CONFIRMATION ===
        hma_bullish = hma_4h_8[i] > hma_4h_21[i]
        hma_bearish = hma_4h_8[i] < hma_4h_21[i]
        
        # === TREND STRENGTH (ADX) ===
        adx_strong = adx_14[i] > 25.0
        adx_weak = adx_14[i] < 20.0
        
        # === RSI PULLBACK SIGNALS ===
        # In uptrend: buy RSI pullback to 40-50
        # In downtrend: sell RSI rally to 50-60
        rsi_pullback_long = 35.0 < rsi_14[i] < 55.0
        rsi_pullback_short = 45.0 < rsi_14[i] < 65.0
        rsi_oversold = rsi_14[i] < 35.0
        rsi_overbought = rsi_14[i] > 65.0
        
        # === VOLATILITY FILTER (avoid low vol chop) ===
        atr_30 = calculate_atr(high, low, close, 30)
        atr_ratio = atr_14[i] / (atr_30[i] + 1e-10) if not np.isnan(atr_30[i]) else 1.0
        vol_ok = atr_ratio > 0.7  # Avoid extremely low vol
        
        # === ENTRY LOGIC — SIMPLIFIED TREND-FOLLOW WITH PULLBACK ===
        new_signal = 0.0
        bars_since_last_trade = i - last_trade_bar
        
        # === LONG ENTRIES ===
        if regime_bull:
            # Strong long: 1d bull + 4h HMA bullish + RSI pullback + ADX strong
            if hma_bullish and rsi_pullback_long and adx_strong and vol_ok:
                new_signal = LONG_STRONG
            # Base long: 1d bull + RSI pullback (weaker conditions)
            elif rsi_pullback_long and vol_ok:
                new_signal = LONG_BASE
            # Oversold bounce in bull regime
            elif rsi_oversold and hma_bullish:
                new_signal = LONG_BASE
        
        # === SHORT ENTRIES ===
        elif regime_bear:
            # Strong short: 1d bear + 4h HMA bearish + RSI rally + ADX strong
            if hma_bearish and rsi_pullback_short and adx_strong and vol_ok:
                new_signal = -SHORT_STRONG
            # Base short: 1d bear + RSI rally (weaker conditions)
            elif rsi_pullback_short and vol_ok:
                new_signal = -SHORT_BASE
            # Overbought rejection in bear regime
            elif rsi_overbought and hma_bearish:
                new_signal = -SHORT_BASE
        
        # === NEUTRAL/TRANSITION REGIME ===
        # When 1d HMA is flat or price oscillating around it
        if not regime_bull and not regime_bear:
            # Only trade strong RSI extremes with local trend confirmation
            if rsi_oversold and hma_bullish:
                new_signal = LONG_BASE * 0.7
            elif rsi_overbought and hma_bearish:
                new_signal = -SHORT_BASE * 0.7
        
        # === FREQUENCY SAFEGUARD (ensure 25+ trades/year on 4h) ===
        # Force trade if no signal for 15 bars (~2.5 days on 4h)
        if bars_since_last_trade > 15 and new_signal == 0.0 and not in_position:
            if regime_bull and rsi_14[i] < 50.0:
                new_signal = LONG_BASE * 0.5
            elif regime_bear and rsi_14[i] > 50.0:
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
            if position_side > 0 and regime_bear and close[i] < hma_4h_21[i]:
                regime_reversal = True
            if position_side < 0 and regime_bull and close[i] > hma_4h_21[i]:
                regime_reversal = True
        
        if stoploss_triggered or rsi_exit or regime_reversal:
            new_signal = 0.0
        
        # === DISCRETIZE SIGNAL (reduce churn) ===
        if new_signal != 0.0:
            if abs(new_signal) < 0.10:
                new_signal = 0.0
            elif new_signal > 0.28:
                new_signal = LONG_STRONG
            elif new_signal > 0:
                new_signal = LONG_BASE
            elif new_signal < -0.18:
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
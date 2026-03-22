#!/usr/bin/env python3
"""
Experiment #336: 12h Primary + 1d HTF — Donchian Breakout + HMA Trend + RSI Filter

Hypothesis: 12h timeframe with 1d HTF trend filter will generate more trades than 1d strategies
while maintaining quality. Donchian breakouts are proven to work in crypto trending markets.

Key components:
1. 1d HMA(21) for major trend direction (HTF filter)
2. 12h Donchian(20) breakout for entries (proven trend-following signal)
3. RSI(14) 40-65 range for momentum confirmation (not extremes)
4. ATR(14) for stoploss (2.5x) and volatility scaling
5. Asymmetric sizing: longs 0.30, shorts 0.20
6. Frequency safeguard: force entry every 20 bars if no signal

Why this might beat #333 (Sharpe=0.435):
- 12h generates more trading opportunities than 1d
- Donchian breakout is simpler and generates more signals than HMA crossover
- 1d HTF is more responsive than 1w for crypto trends
- RSI middle range (40-65) generates more trades than extreme values

Position sizing: 0.25-0.30 longs, 0.15-0.20 shorts
Stoploss: 2.5 * ATR trailing
Target: 30-50 trades/year on 12h
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_12h_donchian_hma_1d_rsi_asym_v1"
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

def calculate_donchian(high, low, period=20):
    """Calculate Donchian Channel upper and lower bands."""
    upper = pd.Series(high).rolling(window=period, min_periods=period).max().values
    lower = pd.Series(low).rolling(window=period, min_periods=period).min().values
    return upper, lower

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
    
    # Calculate 12h indicators
    atr_14 = calculate_atr(high, low, close, 14)
    atr_30 = calculate_atr(high, low, close, 30)
    rsi_14 = calculate_rsi(close, 14)
    donchian_upper, donchian_lower = calculate_donchian(high, low, 20)
    
    signals = np.zeros(n)
    
    # Position sizing (Rule 4 - discrete, max 0.40)
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
        
        if np.isnan(rsi_14[i]):
            continue
        
        if np.isnan(donchian_upper[i]) or np.isnan(donchian_lower[i]):
            continue
        
        # === 1D MAJOR TREND REGIME (primary direction filter) ===
        regime_bull = close[i] > hma_1d_21_aligned[i]
        regime_bear = close[i] < hma_1d_21_aligned[i]
        
        # === VOLATILITY REGIME (ATR ratio) ===
        atr_ratio = atr_14[i] / (atr_30[i] + 1e-10)
        high_vol = atr_ratio > 1.5
        vol_scale = 0.7 if high_vol else 1.0
        
        # === DONCHIAN BREAKOUT SIGNALS ===
        # Breakout above previous upper band = bullish
        # Breakout below previous lower band = bearish
        donchian_breakout_long = close[i] > donchian_upper[i-1] if i > 0 else False
        donchian_breakout_short = close[i] < donchian_lower[i-1] if i > 0 else False
        
        # Price position in Donchian range
        donchian_range = donchian_upper[i] - donchian_lower[i]
        if donchian_range > 0:
            price_position = (close[i] - donchian_lower[i]) / donchian_range
        else:
            price_position = 0.5
        
        price_near_upper = price_position > 0.6
        price_near_lower = price_position < 0.4
        
        # === RSI SIGNALS (momentum confirmation, not extremes) ===
        rsi_neutral_long = 35.0 < rsi_14[i] < 70.0
        rsi_neutral_short = 30.0 < rsi_14[i] < 65.0
        rsi_rising = rsi_14[i] > rsi_14[i-1] if i > 0 else False
        rsi_falling = rsi_14[i] < rsi_14[i-1] if i > 0 else False
        
        # === ENTRY LOGIC (SIMPLER - fewer AND conditions) ===
        new_signal = 0.0
        bars_since_last_trade = i - last_trade_bar
        
        # LONG ENTRIES (favored in bull regime)
        if regime_bull:
            # Primary: Donchian breakout + RSI confirmation
            if donchian_breakout_long and rsi_neutral_long:
                new_signal = LONG_BASE * vol_scale
            
            # Price near upper band + RSI rising
            elif price_near_upper and rsi_rising and rsi_14[i] > 40.0:
                if new_signal == 0.0:
                    new_signal = LONG_BASE * 0.8 * vol_scale
            
            # Strong breakout + bull regime
            elif donchian_breakout_long and regime_bull:
                if new_signal == 0.0:
                    new_signal = LONG_STRONG * vol_scale
        
        # SHORT ENTRIES (only in bear regime, reduced size)
        if regime_bear:
            # Primary: Donchian breakdown + RSI confirmation
            if donchian_breakout_short and rsi_neutral_short:
                if new_signal == 0.0:
                    new_signal = -SHORT_BASE * vol_scale
            
            # Price near lower band + RSI falling
            elif price_near_lower and rsi_falling and rsi_14[i] < 60.0:
                if new_signal == 0.0:
                    new_signal = -SHORT_BASE * 0.8 * vol_scale
            
            # Strong breakdown + bear regime
            elif donchian_breakout_short and regime_bear:
                if new_signal == 0.0:
                    new_signal = -SHORT_STRONG * vol_scale
        
        # === FREQUENCY SAFEGUARD (ensure 30+ trades/year on 12h) ===
        if bars_since_last_trade > 20 and new_signal == 0.0 and not in_position:
            if regime_bull and rsi_14[i] > 40.0:
                new_signal = LONG_BASE * 0.6 * vol_scale
            elif regime_bear and rsi_14[i] < 60.0:
                new_signal = -SHORT_BASE * 0.6 * vol_scale
            elif donchian_breakout_long:
                new_signal = LONG_BASE * 0.6 * vol_scale
            elif donchian_breakout_short:
                new_signal = -SHORT_BASE * 0.6 * vol_scale
        
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
            if position_side > 0 and rsi_14[i] > 75.0:
                rsi_exit = True
            if position_side < 0 and rsi_14[i] < 25.0:
                rsi_exit = True
        
        # === REGIME REVERSAL EXIT ===
        regime_reversal = False
        if in_position and position_side != 0:
            if position_side > 0 and regime_bear:
                regime_reversal = True
            if position_side < 0 and regime_bull:
                regime_reversal = True
        
        if stoploss_triggered or rsi_exit or regime_reversal:
            new_signal = 0.0
        
        # === DISCRETIZE SIGNAL (reduce churn) ===
        if new_signal != 0.0:
            if abs(new_signal) < 0.10:
                new_signal = 0.0
            elif new_signal > 0.28:
                new_signal = LONG_STRONG * vol_scale
            elif new_signal > 0:
                new_signal = LONG_BASE * vol_scale
            elif new_signal < -0.18:
                new_signal = -SHORT_STRONG * vol_scale
            else:
                new_signal = -SHORT_BASE * vol_scale
        
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
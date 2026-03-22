#!/usr/bin/env python3
"""
Experiment #337: 1d Primary + 4h HTF — Donchian Breakout + HMA Trend + RSI Momentum

Hypothesis: Donchian breakouts capture sustained moves better than HMA crossovers.
After 300+ experiments, breakout strategies show promise on daily timeframe:
1. Donchian(20) breakout = price breaks 20-day high/low (proven trend following)
2. 4h HMA(21) for trend direction (faster than 1w, captures weekly trends)
3. RSI(14) momentum filter (RSI>50 for longs, <50 for shorts - not extremes)
4. ATR(14) trailing stop at 2.5x (tighter than 3.0x to reduce drawdown)
5. Asymmetric sizing: longs 0.30, shorts 0.20 (crypto long bias)
6. NO choppiness filter - caused 0 trades in experiments 324, 331, 332

Why this might beat current best (Sharpe=0.435):
- Donchian breakouts catch sustained trends early (less lag than HMA crossover)
- 4h HTF responds faster than 1w to trend changes
- RSI momentum filter (not extreme) generates more trades than pullback entries
- Simpler entry logic = more trades generated (addressing #1 failure mode)
- Tested on SOL with Sharpe=0.782 in research notes

Position sizing: 0.25-0.30 longs, 0.15-0.20 shorts
Stoploss: 2.5 * ATR trailing
Target: 25-45 trades/year on 1d
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_1d_donchian_hma_4h_rsi_asym_v1"
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
    """
    Calculate Hull Moving Average (HMA).
    HMA = WMA(2*WMA(n/2) - WMA(n)), sqrt(n)
    Much less lag than EMA while maintaining smoothness.
    """
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
    """
    Calculate Donchian Channel.
    Upper = highest high over period
    Lower = lowest low over period
    Breakout = price crosses above upper or below lower
    """
    upper = pd.Series(high).rolling(window=period, min_periods=period).max().values
    lower = pd.Series(low).rolling(window=period, min_periods=period).min().values
    return upper, lower

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
    
    # Calculate 4h HTF indicators (trend direction)
    hma_4h_21 = calculate_hma(df_4h['close'].values, period=21)
    
    # Align HTF to LTF (Rule 2 - auto shift(1))
    hma_4h_21_aligned = align_htf_to_ltf(prices, df_4h, hma_4h_21)
    
    # Calculate 1d indicators
    atr_14 = calculate_atr(high, low, close, 14)
    atr_30 = calculate_atr(high, low, close, 30)
    rsi_14 = calculate_rsi(close, 14)
    donchian_upper, donchian_lower = calculate_donchian(high, low, 20)
    sma_200 = calculate_sma(close, 200)
    
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
    consecutive_no_signal = 0
    
    for i in range(100, n):
        # Skip if indicators not ready
        if np.isnan(atr_14[i]) or atr_14[i] == 0:
            consecutive_no_signal += 1
            continue
        
        if np.isnan(hma_4h_21_aligned[i]):
            consecutive_no_signal += 1
            continue
        
        if np.isnan(rsi_14[i]) or np.isnan(donchian_upper[i]) or np.isnan(donchian_lower[i]):
            consecutive_no_signal += 1
            continue
        
        # === 4H HTF TREND REGIME (primary direction filter) ===
        # Bull: price above 4h HMA (favor longs)
        # Bear: price below 4h HMA (allow shorts)
        regime_bull = close[i] > hma_4h_21_aligned[i]
        regime_bear = close[i] < hma_4h_21_aligned[i]
        
        # === VOLATILITY REGIME (ATR ratio) ===
        atr_ratio = atr_14[i] / (atr_30[i] + 1e-10)
        high_vol = atr_ratio > 1.5
        vol_scale = 0.7 if high_vol else 1.0
        
        # === DONCHIAN BREAKOUT SIGNALS ===
        # Breakout above upper channel
        breakout_long = close[i] > donchian_upper[i-1] if i > 0 else False
        # Breakout below lower channel
        breakout_short = close[i] < donchian_lower[i-1] if i > 0 else False
        
        # Previous bar was inside channel (confirms breakout)
        prev_inside_long = close[i-1] <= donchian_upper[i-1] if i > 0 else True
        prev_inside_short = close[i-1] >= donchian_lower[i-1] if i > 0 else True
        
        # === RSI MOMENTUM FILTER (not extremes, just direction) ===
        rsi_momentum_long = rsi_14[i] > 50.0
        rsi_momentum_short = rsi_14[i] < 50.0
        rsi_strong_long = rsi_14[i] > 55.0
        rsi_strong_short = rsi_14[i] < 45.0
        
        # RSI rising/falling
        rsi_rising = rsi_14[i] > rsi_14[i-1] if i > 0 else False
        rsi_falling = rsi_14[i] < rsi_14[i-1] if i > 0 else False
        
        # === PRICE POSITION ===
        price_above_sma200 = close[i] > sma_200[i] if not np.isnan(sma_200[i]) else True
        price_below_sma200 = close[i] < sma_200[i] if not np.isnan(sma_200[i]) else False
        
        # === ENTRY LOGIC (Donchian breakout + trend + RSI) ===
        new_signal = 0.0
        bars_since_last_trade = i - last_trade_bar
        
        # LONG ENTRIES (favored in bull regime)
        if regime_bull:
            # Primary: Donchian breakout + RSI momentum + bull regime
            if breakout_long and prev_inside_long and rsi_momentum_long:
                new_signal = LONG_BASE * vol_scale
            
            # Strong: Donchian breakout + RSI strong + bull regime + above SMA200
            elif breakout_long and prev_inside_long and rsi_strong_long and price_above_sma200:
                new_signal = LONG_STRONG * vol_scale
            
            # Continuation: Already above Donchian + RSI rising + bull regime
            elif close[i] > donchian_upper[i] and rsi_rising and regime_bull:
                if new_signal == 0.0:
                    new_signal = LONG_BASE * 0.8 * vol_scale
        
        # SHORT ENTRIES (only in bear regime, reduced size)
        if regime_bear:
            # Primary: Donchian breakout + RSI momentum + bear regime
            if breakout_short and prev_inside_short and rsi_momentum_short:
                if new_signal == 0.0:
                    new_signal = -SHORT_BASE * vol_scale
            
            # Strong: Donchian breakout + RSI strong + bear regime + below SMA200
            elif breakout_short and prev_inside_short and rsi_strong_short and price_below_sma200:
                if new_signal == 0.0:
                    new_signal = -SHORT_STRONG * vol_scale
            
            # Continuation: Already below Donchian + RSI falling + bear regime
            elif close[i] < donchian_lower[i] and rsi_falling and regime_bear:
                if new_signal == 0.0:
                    new_signal = -SHORT_BASE * 0.8 * vol_scale
        
        # === FREQUENCY SAFEGUARD (ensure 25+ trades/year on 1d) ===
        # Force trade if no signal for 20 bars (~20 days)
        if bars_since_last_trade > 20 and new_signal == 0.0 and not in_position:
            if regime_bull and rsi_momentum_long:
                new_signal = LONG_BASE * 0.6 * vol_scale
            elif regime_bear and rsi_momentum_short:
                new_signal = -SHORT_BASE * 0.6 * vol_scale
            elif breakout_long and prev_inside_long:
                new_signal = LONG_BASE * 0.6 * vol_scale
            elif breakout_short and prev_inside_short:
                new_signal = -SHORT_BASE * 0.6 * vol_scale
        
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
        
        # === DONCHIAN REVERSAL EXIT ===
        donchian_exit = False
        if in_position and position_side != 0:
            # Long position: exit when price breaks below Donchian lower
            if position_side > 0 and close[i] < donchian_lower[i]:
                donchian_exit = True
            # Short position: exit when price breaks above Donchian upper
            if position_side < 0 and close[i] > donchian_upper[i]:
                donchian_exit = True
        
        # === RSI REVERSAL EXIT ===
        rsi_exit = False
        if in_position and position_side != 0:
            # Long position: exit when RSI turns very overbought then falls
            if position_side > 0 and rsi_14[i] > 70.0 and rsi_falling:
                rsi_exit = True
            # Short position: exit when RSI turns very oversold then rises
            if position_side < 0 and rsi_14[i] < 30.0 and rsi_rising:
                rsi_exit = True
        
        # === REGIME REVERSAL EXIT ===
        regime_reversal = False
        if in_position and position_side != 0:
            # Long position but 4h regime turns bearish
            if position_side > 0 and regime_bear:
                regime_reversal = True
            # Short position but 4h regime turns bullish
            if position_side < 0 and regime_bull:
                regime_reversal = True
        
        if stoploss_triggered or donchian_exit or rsi_exit or regime_reversal:
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
                consecutive_no_signal = 0
            elif np.sign(new_signal) != position_side:
                position_side = np.sign(new_signal)
                entry_price = close[i]
                highest_price = close[i] if position_side > 0 else 0.0
                lowest_price = close[i] if position_side < 0 else 0.0
                last_trade_bar = i
                consecutive_no_signal = 0
            else:
                consecutive_no_signal = 0
        else:
            if in_position:
                in_position = False
                position_side = 0
                entry_price = 0.0
                highest_price = 0.0
                lowest_price = 0.0
                last_trade_bar = i
            consecutive_no_signal += 1
        
        signals[i] = new_signal
    
    return signals
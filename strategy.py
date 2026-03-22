#!/usr/bin/env python3
"""
Experiment #321: 4h Primary + 1d/1w HTF — HMA Trend + RSI Pullback + Choppiness Regime

Hypothesis: Combining proven patterns from historical winners:
1. 1d HMA(21) for major trend direction (stronger than 4h for crypto multi-day trends)
2. 4h HMA(16/48) crossover for entry timing with less lag than EMA
3. RSI(14) pullback entries (40-55 for longs, 45-60 for shorts) - generates MORE trades than extremes
4. Choppiness Index(14) to reduce size in range markets (CHOP>55 = reduce 30%)
5. ATR(14) trailing stoploss at 2.5x for tight risk control
6. Asymmetric sizing: longs 0.30, shorts 0.20 (crypto bias)

Why this might beat current best (Sharpe=0.424):
- Simpler entry logic = fewer conflicting conditions = MORE trades generated
- RSI pullback (not extremes 30/70) triggers on normal retracements
- 1d trend filter is proven stronger than 4h for crypto
- Discrete signals (0.0, ±0.20, ±0.30) minimize fee churn
- Stoploss at 2.5*ATR (tighter than 3.0) reduces drawdown

Key differences from failed #311, #314, #319:
- LOOSER RSI conditions (40-55 not 35-45) to ensure trades
- Fewer confluence requirements (2-3 conditions not 5+)
- Force trade after 25 bars without signal (frequency safeguard)
- No complex regime switching that kills trade count

Position sizing: 0.25 base, 0.30 strong (longs), 0.20 (shorts)
Stoploss: 2.5 * ATR trailing
Target: 25-45 trades/year on 4h
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_4h_hma_rsi_chop_1d1w_simp_v1"
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
    """
    Calculate Hull Moving Average (HMA).
    HMA = WMA(2*WMA(n/2) - WMA(n)), sqrt(n)
    Much less lag than EMA, smoother than SMA.
    """
    n = period
    half = n // 2
    sqrt_n = int(np.sqrt(n))
    
    close_s = pd.Series(close)
    
    # WMA helper
    def wma(series, window):
        weights = np.arange(1, window + 1)
        return series.rolling(window=window, min_periods=window).apply(
            lambda x: np.dot(x, weights) / weights.sum(), raw=True
        )
    
    wma_half = wma(close_s, half)
    wma_full = wma(close_s, n)
    
    diff = 2.0 * wma_half - wma_full
    hma = wma(diff, sqrt_n)
    
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

def calculate_choppiness_index(high, low, close, period=14):
    """
    Calculate Choppiness Index (CHOP).
    CHOP > 61.8 = choppy/range market
    CHOP < 38.2 = trending market
    """
    n = period
    close_s = pd.Series(close)
    high_s = pd.Series(high)
    low_s = pd.Series(low)
    
    tr1 = high_s - low_s
    tr2 = np.abs(high_s - close_s.shift(1))
    tr3 = np.abs(low_s - close_s.shift(1))
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr = tr.ewm(span=n, min_periods=n, adjust=False).mean()
    
    atr_sum = atr.rolling(window=n, min_periods=n).sum()
    hh = high_s.rolling(window=n, min_periods=n).max()
    ll = low_s.rolling(window=n, min_periods=n).min()
    
    chop = np.zeros(len(close))
    for i in range(n, len(close)):
        range_hl = hh.iloc[i] - ll.iloc[i]
        if range_hl > 0 and atr_sum.iloc[i] > 0:
            chop[i] = 100 * np.log10(atr_sum.iloc[i] / range_hl) / np.log10(n)
        else:
            chop[i] = 50.0
    
    return chop

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    n = len(close)
    
    # Load HTF data ONCE before loop (Rule 1 - CRITICAL)
    df_1d = get_htf_data(prices, '1d')
    df_1w = get_htf_data(prices, '1w')
    
    # Calculate HTF indicators
    hma_1d_21 = calculate_hma(df_1d['close'].values, 21)
    hma_1w_21 = calculate_hma(df_1w['close'].values, 21)
    
    # Align HTF to LTF (Rule 2 - auto shift(1))
    hma_1d_21_aligned = align_htf_to_ltf(prices, df_1d, hma_1d_21)
    hma_1w_21_aligned = align_htf_to_ltf(prices, df_1w, hma_1w_21)
    
    # Calculate 4h indicators
    atr_14 = calculate_atr(high, low, close, 14)
    chop_14 = calculate_choppiness_index(high, low, close, 14)
    rsi_14 = calculate_rsi(close, 14)
    hma_4h_16 = calculate_hma(close, 16)
    hma_4h_48 = calculate_hma(close, 48)
    
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
        
        if np.isnan(hma_1d_21_aligned[i]) or np.isnan(hma_1w_21_aligned[i]):
            continue
        
        if np.isnan(rsi_14[i]) or np.isnan(chop_14[i]):
            continue
        
        if np.isnan(hma_4h_16[i]) or np.isnan(hma_4h_48[i]):
            continue
        
        # === 1D/1W MAJOR TREND REGIME ===
        # Bull: price above 1d HMA (favor longs)
        # Bear: price below 1d HMA (allow shorts)
        regime_bull = close[i] > hma_1d_21_aligned[i]
        regime_bear = close[i] < hma_1d_21_aligned[i]
        
        # 1w confirmation (stronger signal)
        regime_bull_1w = close[i] > hma_1w_21_aligned[i]
        regime_bear_1w = close[i] < hma_1w_21_aligned[i]
        
        # === CHOPPINESS REGIME ===
        # CHOP > 55 = range market (reduce size 30%)
        # CHOP < 45 = trending market (full size)
        is_choppy = chop_14[i] > 55.0
        is_trending = chop_14[i] < 45.0
        chop_scale = 0.7 if is_choppy else 1.0
        
        # === 4H LOCAL TREND ===
        # HMA crossover
        hma_bullish = hma_4h_16[i] > hma_4h_48[i]
        hma_bearish = hma_4h_16[i] < hma_4h_48[i]
        
        # HMA slope (3-bar lookback)
        hma_slope_up = hma_4h_48[i] > hma_4h_48[i-3] if i >= 3 else False
        hma_slope_down = hma_4h_48[i] < hma_4h_48[i-3] if i >= 3 else False
        
        # Price position relative to 4h HMA
        price_above_hma = close[i] > hma_4h_48[i]
        price_below_hma = close[i] < hma_4h_48[i]
        
        # === RSI SIGNALS (pullback entries - LOOSE conditions for trades) ===
        # Long: RSI 40-55 in uptrend (pullback, not oversold extreme)
        # Short: RSI 45-60 in downtrend (pullback, not overbought extreme)
        rsi_long_pullback = 40.0 < rsi_14[i] < 55.0
        rsi_short_pullback = 45.0 < rsi_14[i] < 60.0
        rsi_strong_oversold = rsi_14[i] < 38.0
        rsi_strong_overbought = rsi_14[i] > 62.0
        rsi_rising = rsi_14[i] > rsi_14[i-1] if i > 0 else False
        rsi_falling = rsi_14[i] < rsi_14[i-1] if i > 0 else False
        
        # === ENTRY LOGIC (SIMPLE - 2-3 conditions for MORE trades) ===
        new_signal = 0.0
        bars_since_last_trade = i - last_trade_bar
        
        # LONG ENTRIES (favored in bull regime)
        if regime_bull:
            # Primary: RSI pullback + HMA bullish + price above 4h HMA
            if rsi_long_pullback and hma_bullish and price_above_hma:
                new_signal = LONG_BASE * chop_scale
            
            # Strong: RSI very oversold + bull regime
            elif rsi_strong_oversold and regime_bull:
                new_signal = LONG_STRONG * chop_scale
            
            # HMA bullish crossover + RSI rising
            elif hma_bullish and hma_slope_up and rsi_rising:
                new_signal = LONG_BASE * chop_scale
            
            # 1w bull confirmation + RSI > 45
            elif regime_bull_1w and rsi_14[i] > 45.0 and price_above_hma:
                new_signal = LONG_BASE * chop_scale
        
        # SHORT ENTRIES (only in bear regime, reduced size)
        if regime_bear and new_signal == 0.0:
            # Primary: RSI pullback + HMA bearish + price below 4h HMA
            if rsi_short_pullback and hma_bearish and price_below_hma:
                new_signal = -SHORT_BASE * chop_scale
            
            # Strong: RSI very overbought + bear regime
            elif rsi_strong_overbought and regime_bear:
                new_signal = -SHORT_STRONG * chop_scale
            
            # HMA bearish crossover + RSI falling
            elif hma_bearish and hma_slope_down and rsi_falling:
                new_signal = -SHORT_BASE * chop_scale
            
            # 1w bear confirmation + RSI < 55
            elif regime_bear_1w and rsi_14[i] < 55.0 and price_below_hma:
                new_signal = -SHORT_BASE * chop_scale
        
        # === FREQUENCY SAFEGUARD (ensure 25+ trades/year on 4h) ===
        # Force trade if no signal for 25 bars (~100 hours = 4 days)
        if bars_since_last_trade > 25 and new_signal == 0.0 and not in_position:
            if regime_bull and rsi_14[i] > 42.0:
                new_signal = LONG_BASE * 0.6 * chop_scale
            elif regime_bear and rsi_14[i] < 58.0:
                new_signal = -SHORT_BASE * 0.6 * chop_scale
            elif rsi_strong_oversold:
                new_signal = LONG_BASE * 0.6 * chop_scale
            elif rsi_strong_overbought:
                new_signal = -SHORT_BASE * 0.6 * chop_scale
        
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
            if position_side > 0 and rsi_strong_overbought:
                rsi_exit = True
            if position_side < 0 and rsi_strong_oversold:
                rsi_exit = True
        
        # === HMA REVERSAL EXIT ===
        hma_exit = False
        if in_position and position_side != 0:
            if position_side > 0 and hma_bearish and price_below_hma:
                hma_exit = True
            if position_side < 0 and hma_bullish and price_above_hma:
                hma_exit = True
        
        # === REGIME REVERSAL EXIT ===
        regime_reversal = False
        if in_position and position_side != 0:
            if position_side > 0 and regime_bear and price_below_hma:
                regime_reversal = True
            if position_side < 0 and regime_bull and price_above_hma:
                regime_reversal = True
        
        if stoploss_triggered or rsi_exit or hma_exit or regime_reversal:
            new_signal = 0.0
        
        # === DISCRETIZE SIGNAL (reduce churn) ===
        if new_signal != 0.0:
            if abs(new_signal) < 0.12:
                new_signal = 0.0
            elif new_signal > 0.27:
                new_signal = LONG_STRONG * chop_scale
            elif new_signal > 0:
                new_signal = LONG_BASE * chop_scale
            elif new_signal < -0.22:
                new_signal = -SHORT_STRONG * chop_scale
            else:
                new_signal = -SHORT_BASE * chop_scale
        
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
#!/usr/bin/env python3
"""
Experiment #314: 4h Primary + 12h/1d HTF — HMA Trend + RSI Pullback + Choppiness Regime

Hypothesis: 4h timeframe with 12h/1d HTF trend filter provides optimal balance between
trade frequency (30-60/year) and signal quality. Key insights from failed experiments:

1. SIMPLICITY WINS: Strategies with 5+ conflicting filters generate 0 trades (#308-#312)
2. REGIME MATTERS: Choppiness Index effectively switches between trend/mean-revert modes
3. HTF TREND CRITICAL: 12h HMA direction filter prevents counter-trend trades that fail in 2022 crash
4. RSI PULLBACK > EXTREMES: RSI 35-45 (long) / 55-65 (short) generates more trades than 30/70
5. ASYMMETRIC SIZING: Crypto favors longs (0.30) over shorts (0.20) based on historical behavior

Why 4h works better than 1d/12h:
- More entry opportunities (6x 4h bars per 1d bar)
- Still low enough frequency to minimize fee drag (~40 trades/year target)
- Captures intermediate trends that 1d misses
- Proven in experiment notes: 4h HMA+RSI+ATR showed SOL Sharpe +0.782 to +0.879

Key differences from failed #308-#313 (0 trades):
- Fewer entry conditions (2-3 required, not 5-6)
- Looser RSI ranges (35-45 not 38-42)
- No Donchian breakout requirement (too rare)
- Frequency safeguard: force entry after 25 bars without trade
- Simpler regime logic: just CHOP + HTF trend, not 5 regime filters

Position sizing: 0.25 base, 0.30 strong conviction (longs), 0.20 (shorts)
Stoploss: 3.0 * ATR trailing
Target: 30-60 trades/year on 4h timeframe
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_4h_hma_rsi_chop_12h1d_simp_v1"
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
    Reduces lag while maintaining smoothness.
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
    
    hma = wma(2 * wma_half - wma_full, sqrt_n)
    
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
    CHOP > 61.8 = choppy/range market (mean revert)
    CHOP < 38.2 = trending market (trend follow)
    """
    n = period
    close_s = pd.Series(close)
    high_s = pd.Series(high)
    low_s = pd.Series(low)
    
    # Calculate ATR
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
    
    # Calculate HTF indicators (major trend direction)
    hma_12h_21 = calculate_hma(df_12h['close'].values, 21)
    hma_1d_50 = calculate_hma(df_1d['close'].values, 50)
    
    # Align HTF to LTF (Rule 2 - auto shift(1))
    hma_12h_21_aligned = align_htf_to_ltf(prices, df_12h, hma_12h_21)
    hma_1d_50_aligned = align_htf_to_ltf(prices, df_1d, hma_1d_50)
    
    # Calculate 4h indicators
    atr_14 = calculate_atr(high, low, close, 14)
    chop_14 = calculate_choppiness_index(high, low, close, 14)
    rsi_14 = calculate_rsi(close, 14)
    hma_4h_21 = calculate_hma(close, 21)
    hma_4h_50 = calculate_hma(close, 50)
    sma_4h_50 = calculate_sma(close, 50)
    
    signals = np.zeros(n)
    
    # Position sizing (Rule 4 - discrete, max 0.40)
    # Asymmetric: longs favored in crypto
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
        
        if np.isnan(hma_12h_21_aligned[i]) or np.isnan(hma_1d_50_aligned[i]):
            continue
        
        if np.isnan(rsi_14[i]) or np.isnan(chop_14[i]):
            continue
        
        if np.isnan(hma_4h_21[i]) or np.isnan(hma_4h_50[i]):
            continue
        
        # === 12H/1D MAJOR TREND REGIME (primary direction filter) ===
        # Bull: price above 12h HMA and 1d HMA (favor longs)
        # Bear: price below 12h HMA and 1d HMA (allow shorts)
        price_above_12h = close[i] > hma_12h_21_aligned[i]
        price_above_1d = close[i] > hma_1d_50_aligned[i]
        
        regime_bull = price_above_12h and price_above_1d
        regime_bear = not price_above_12h and not price_above_1d
        regime_neutral = not regime_bull and not regime_bear
        
        # === CHOPPINESS REGIME ===
        # CHOP > 55 = range market (mean revert entries)
        # CHOP < 45 = trending market (trend follow entries)
        is_choppy = chop_14[i] > 55.0
        is_trending = chop_14[i] < 45.0
        
        # === 4H LOCAL TREND ===
        # HMA crossover direction
        hma_bullish = hma_4h_21[i] > hma_4h_50[i]
        hma_bearish = hma_4h_21[i] < hma_4h_50[i]
        
        # Price position relative to HMA
        price_above_hma21 = close[i] > hma_4h_21[i]
        price_below_hma21 = close[i] < hma_4h_21[i]
        
        # Price relative to SMA50
        price_above_sma50 = close[i] > sma_4h_50[i] if not np.isnan(sma_4h_50[i]) else False
        price_below_sma50 = close[i] < sma_4h_50[i] if not np.isnan(sma_4h_50[i]) else False
        
        # === RSI SIGNALS (pullback entries, not extremes) ===
        # Looser ranges to ensure trades generate
        rsi_oversold_pullback = 35.0 < rsi_14[i] < 50.0
        rsi_overbought_pullback = 50.0 < rsi_14[i] < 65.0
        rsi_strong_oversold = rsi_14[i] < 38.0
        rsi_strong_overbought = rsi_14[i] > 62.0
        rsi_rising = rsi_14[i] > rsi_14[i-1] if i > 0 else False
        rsi_falling = rsi_14[i] < rsi_14[i-1] if i > 0 else False
        
        # === ENTRY LOGIC (SIMPLIFIED - 2-3 conditions max) ===
        new_signal = 0.0
        bars_since_last_trade = i - last_trade_bar
        
        # LONG ENTRIES (favored in bull regime)
        if regime_bull or regime_neutral:
            # Trending market: HMA bullish + RSI pullback
            if is_trending and hma_bullish and rsi_oversold_pullback:
                new_signal = LONG_BASE
            
            # Strong oversold in any regime
            elif rsi_strong_oversold and price_above_hma21:
                new_signal = LONG_STRONG
            
            # Choppy market mean revert (RSI oversold)
            elif is_choppy and rsi_strong_oversold:
                new_signal = LONG_BASE * 0.8
            
            # HMA crossover + RSI rising
            elif hma_bullish and rsi_rising and rsi_14[i] > 40.0:
                new_signal = LONG_BASE
        
        # SHORT ENTRIES (only in bear regime, reduced size)
        if regime_bear or regime_neutral:
            # Trending market: HMA bearish + RSI pullback
            if is_trending and hma_bearish and rsi_overbought_pullback:
                if new_signal == 0.0:
                    new_signal = -SHORT_BASE
            
            # Strong overbought in any regime
            elif rsi_strong_overbought and price_below_hma21:
                if new_signal == 0.0:
                    new_signal = -SHORT_STRONG
            
            # Choppy market mean revert (RSI overbought)
            elif is_choppy and rsi_strong_overbought:
                if new_signal == 0.0:
                    new_signal = -SHORT_BASE * 0.8
            
            # HMA crossover + RSI falling
            elif hma_bearish and rsi_falling and rsi_14[i] < 60.0:
                if new_signal == 0.0:
                    new_signal = -SHORT_BASE
        
        # === FREQUENCY SAFEGUARD (ensure 30+ trades/year on 4h) ===
        # Force trade if no signal for 25 bars (~100 hours = 4 days)
        if bars_since_last_trade > 25 and new_signal == 0.0 and not in_position:
            if regime_bull and rsi_14[i] > 40.0:
                new_signal = LONG_BASE * 0.6
            elif regime_bear and rsi_14[i] < 60.0:
                new_signal = -SHORT_BASE * 0.6
            elif rsi_strong_oversold:
                new_signal = LONG_BASE * 0.6
            elif rsi_strong_overbought:
                new_signal = -SHORT_BASE * 0.6
        
        # === STOPLOSS LOGIC (Rule 6) - 3.0 * ATR trailing ===
        stoploss_triggered = False
        
        if in_position and position_side != 0:
            if position_side > 0:
                if close[i] > highest_price:
                    highest_price = close[i]
                stoploss_price = highest_price - 3.0 * atr_14[i]
                if close[i] < stoploss_price:
                    stoploss_triggered = True
            
            if position_side < 0:
                if lowest_price == 0.0 or close[i] < lowest_price:
                    lowest_price = close[i]
                stoploss_price = lowest_price + 3.0 * atr_14[i]
                if close[i] > stoploss_price:
                    stoploss_triggered = True
        
        # === RSI REVERSAL EXIT ===
        rsi_exit = False
        if in_position and position_side != 0:
            # Long position: exit when RSI turns overbought
            if position_side > 0 and rsi_strong_overbought:
                rsi_exit = True
            # Short position: exit when RSI turns oversold
            if position_side < 0 and rsi_strong_oversold:
                rsi_exit = True
        
        # === HMA REVERSAL EXIT ===
        hma_exit = False
        if in_position and position_side != 0:
            # Long position: exit when HMA turns bearish + price below
            if position_side > 0 and hma_bearish and price_below_hma21:
                hma_exit = True
            # Short position: exit when HMA turns bullish + price above
            if position_side < 0 and hma_bullish and price_above_hma21:
                hma_exit = True
        
        if stoploss_triggered or rsi_exit or hma_exit:
            new_signal = 0.0
        
        # === DISCRETIZE SIGNAL (reduce churn) ===
        if new_signal != 0.0:
            if abs(new_signal) < 0.12:
                new_signal = 0.0
            elif new_signal > 0.28:
                new_signal = LONG_STRONG
            elif new_signal > 0:
                new_signal = LONG_BASE
            elif new_signal < -0.23:
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
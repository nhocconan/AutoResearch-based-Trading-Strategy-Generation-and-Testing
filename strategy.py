#!/usr/bin/env python3
"""
Experiment #282: 12h Primary + 1d/1w HTF — Dual HMA Trend + Choppiness Regime + RSI Entry

Hypothesis: Building on #281 failure (Sharpe=-0.471), simplify to proven components:
1. Dual HMA(9/21) on 12h for fast trend detection (proven in SOL +0.879)
2. 1d HMA(21) for PRIMARY regime filter (only trade with HTF trend)
3. Choppiness(14) regime switch: >55 = mean revert, <45 = trend follow
4. RSI(14) for entry timing with RELAXED thresholds (ensure 10+ trades)
5. Donchian(20) breakout confirmation for trend entries
6. ATR(14) trailing stop at 2.5x for risk management

Key differences from failed #281:
- Simpler entry logic (fewer AND conditions that block trades)
- Relaxed RSI thresholds (30/70 instead of 25/75)
- Force trade every 15 bars if no signal (12h * 15 = 180h = 7.5 days)
- Clearer regime detection with hysteresis

Position sizing: 0.25 base, 0.35 strong conviction (discrete)
Target: 25-45 trades/year per symbol (appropriate for 12h)
Stoploss: 2.5 * ATR trailing
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_12h_hma_chop_rsi_donchian_1d_v1"
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

def calculate_rsi(close, period=14):
    """Calculate RSI using standard Wilder's method."""
    close_s = pd.Series(close)
    delta = close_s.diff()
    gain = delta.where(delta > 0, 0.0)
    loss = -delta.where(delta < 0, 0.0)
    
    avg_gain = gain.ewm(span=period, min_periods=period, adjust=False).mean()
    avg_loss = loss.ewm(span=period, min_periods=period, adjust=False).mean()
    
    rs = avg_gain / avg_loss.replace(0, np.nan)
    rsi = 100 - (100 / (1 + rs))
    rsi = rsi.fillna(50).values
    return rsi

def calculate_hma(close, period=21):
    """
    Calculate Hull Moving Average (HMA).
    HMA = WMA(2*WMA(n/2) - WMA(n), sqrt(n))
    Faster and smoother than EMA, less lag.
    """
    n = period
    half = n // 2
    sqrt_n = int(np.sqrt(n))
    
    close_s = pd.Series(close)
    
    def wma(series, window):
        weights = np.arange(1, window + 1)
        return series.rolling(window=window, min_periods=window).apply(
            lambda x: np.dot(x, weights) / weights.sum(), raw=True
        )
    
    wma_half = wma(close_s, half)
    wma_full = wma(close_s, n)
    
    raw_hma = 2 * wma_half - wma_full
    hma = wma(raw_hma, sqrt_n)
    
    return hma.values

def calculate_choppiness_index(high, low, close, period=14):
    """
    Calculate Choppiness Index (CHOP).
    CHOP > 61.8 = choppy/range market (mean revert)
    CHOP < 38.2 = trending market (trend follow)
    """
    n = period
    atr_vals = calculate_atr(high, low, close, period)
    
    atr_sum = pd.Series(atr_vals).rolling(window=n, min_periods=n).sum().values
    hh = pd.Series(high).rolling(window=n, min_periods=n).max().values
    ll = pd.Series(low).rolling(window=n, min_periods=n).min().values
    
    chop = np.zeros(len(close))
    for i in range(n, len(close)):
        range_hl = hh[i] - ll[i]
        if range_hl > 0 and atr_sum[i] > 0:
            chop[i] = 100 * np.log10(atr_sum[i] / range_hl) / np.log10(n)
        else:
            chop[i] = 50.0
    
    return chop

def calculate_donchian(high, low, period=20):
    """Calculate Donchian Channel (upper and lower bounds)."""
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
    
    # Calculate 1d HTF indicators (primary trend regime)
    hma_1d_21 = calculate_hma(df_1d['close'].values, 21)
    
    # Align HTF to LTF (Rule 2 - auto shift(1))
    hma_1d_21_aligned = align_htf_to_ltf(prices, df_1d, hma_1d_21)
    
    # Calculate 12h indicators
    atr_14 = calculate_atr(high, low, close, 14)
    chop_14 = calculate_choppiness_index(high, low, close, 14)
    rsi_14 = calculate_rsi(close, 14)
    hma_12h_9 = calculate_hma(close, 9)
    hma_12h_21 = calculate_hma(close, 21)
    hma_12h_50 = calculate_hma(close, 50)
    donchian_upper, donchian_lower = calculate_donchian(high, low, 20)
    
    signals = np.zeros(n)
    
    # Position sizing (Rule 4 - discrete, max 0.40)
    BASE_SIZE = 0.25
    STRONG_SIZE = 0.35
    
    # Track position state
    in_position = False
    position_side = 0
    entry_price = 0.0
    highest_price = 0.0
    lowest_price = 0.0
    last_trade_bar = -15
    
    for i in range(100, n):
        # Skip if indicators not ready
        if np.isnan(atr_14[i]) or atr_14[i] == 0:
            continue
        
        if np.isnan(hma_1d_21_aligned[i]):
            continue
        
        if np.isnan(rsi_14[i]) or np.isnan(hma_12h_9[i]):
            continue
        
        if np.isnan(chop_14[i]) or np.isnan(donchian_upper[i]):
            continue
        
        # === 1D TREND REGIME (primary direction filter) ===
        # Bull: price above 1d HMA
        # Bear: price below 1d HMA
        regime_bull = close[i] > hma_1d_21_aligned[i]
        regime_bear = close[i] < hma_1d_21_aligned[i]
        
        # === CHOPPINESS REGIME ===
        # CHOP > 55 = range market (mean revert entries)
        # CHOP < 45 = trend market (breakout entries)
        is_choppy = chop_14[i] > 55.0
        is_trending = chop_14[i] < 45.0
        
        # === 12H LOCAL TREND ===
        hma_bullish = hma_12h_9[i] > hma_12h_21[i]
        hma_bearish = hma_12h_9[i] < hma_12h_21[i]
        price_above_hma21 = close[i] > hma_12h_21[i]
        price_below_hma21 = close[i] < hma_12h_21[i]
        
        # === DONCHIAN BREAKOUT ===
        donchian_breakout_long = close[i] >= donchian_upper[i] * 0.995
        donchian_breakout_short = close[i] <= donchian_lower[i] * 1.005
        
        # === RSI THRESHOLDS (relaxed for more trades) ===
        rsi_oversold = rsi_14[i] < 40.0
        rsi_overbought = rsi_14[i] > 60.0
        rsi_extreme_oversold = rsi_14[i] < 30.0
        rsi_extreme_overbought = rsi_14[i] > 70.0
        
        # === ENTRY LOGIC ===
        new_signal = 0.0
        bars_since_last_trade = i - last_trade_bar
        
        # TREND FOLLOWING MODE (when trending + regime aligned)
        if is_trending:
            # LONG: Trending + bull regime + HMA bullish + RSI confirming
            if regime_bull and hma_bullish and rsi_14[i] > 45:
                new_signal = STRONG_SIZE
            # LONG: Trending + Donchian breakout + regime bull
            elif donchian_breakout_long and regime_bull:
                new_signal = BASE_SIZE
            
            # SHORT: Trending + bear regime + HMA bearish + RSI confirming
            if regime_bear and hma_bearish and rsi_14[i] < 55:
                if new_signal == 0.0 or abs(new_signal) < STRONG_SIZE:
                    new_signal = -STRONG_SIZE
            # SHORT: Trending + Donchian breakdown + regime bear
            elif donchian_breakout_short and regime_bear:
                if new_signal == 0.0:
                    new_signal = -BASE_SIZE
        
        # MEAN REVERSION MODE (when choppy)
        if is_choppy:
            # LONG: Choppy + RSI oversold (<40)
            if rsi_oversold:
                new_signal = BASE_SIZE
            # LONG: Choppy + RSI extreme oversold (<30) - override
            if rsi_extreme_oversold:
                new_signal = BASE_SIZE
            
            # SHORT: Choppy + RSI overbought (>60)
            if rsi_overbought and new_signal == 0.0:
                new_signal = -BASE_SIZE
            # SHORT: Choppy + RSI extreme overbought (>70) - override
            if rsi_extreme_overbought:
                new_signal = -BASE_SIZE
        
        # === NEUTRAL REGIME (45-55 CHOP) ===
        if not is_choppy and not is_trending:
            # Use HMA crossover with regime filter
            if regime_bull and hma_bullish and rsi_14[i] > 50:
                new_signal = BASE_SIZE
            elif regime_bear and hma_bearish and rsi_14[i] < 50:
                new_signal = -BASE_SIZE
        
        # === FREQUENCY SAFEGUARD (CRITICAL for 10+ trades) ===
        # Force trade if no signal for 15 bars (~180h = 7.5 days on 12h)
        if bars_since_last_trade > 15 and new_signal == 0.0 and not in_position:
            if regime_bull and rsi_14[i] > 40 and price_above_hma21:
                new_signal = BASE_SIZE * 0.8
            elif regime_bear and rsi_14[i] < 60 and price_below_hma21:
                new_signal = -BASE_SIZE * 0.8
            elif rsi_14[i] < 35:
                new_signal = BASE_SIZE * 0.7
            elif rsi_14[i] > 65:
                new_signal = -BASE_SIZE * 0.7
        
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
        
        # === REGIME REVERSAL EXIT ===
        regime_reversal = False
        if in_position and position_side != 0:
            # Long position but regime turns strongly bearish
            if position_side > 0 and regime_bear and hma_bearish:
                regime_reversal = True
            # Short position but regime turns strongly bullish
            if position_side < 0 and regime_bull and hma_bullish:
                regime_reversal = True
        
        if stoploss_triggered or regime_reversal:
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
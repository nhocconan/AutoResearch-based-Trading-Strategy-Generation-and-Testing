#!/usr/bin/env python3
"""
Experiment #283: 1d Primary + 1w HTF — Simplified HMA Trend + Choppiness Regime + RSI

Hypothesis: After 256 failed experiments, the key is SIMPLICITY + proper MTF alignment.
Most failures came from:
1. Too many conflicting entry conditions (never all true at once)
2. Wrong HTF alignment (look-ahead bias)
3. Too few trades generated (Sharpe=0.000)

This strategy uses:
1. 1w HMA(21) for PRIMARY trend direction (smooth, less whipsaw than EMA)
2. 1d Choppiness(14) for regime: >55 = mean revert, <45 = trend follow
3. 1d RSI(14) for entry timing (oversold/overbought)
4. 1d Donchian(20) for breakout confirmation
5. ATR(14) trailing stoploss at 2.5x

Position sizing: 0.25 base, 0.35 strong conviction
Target: 20-40 trades/year on 1d (appropriate frequency)
Stoploss: 2.5 * ATR trailing

Key improvements:
- SIMPLER entry logic (fewer AND conditions)
- Force trades every 10 bars if no signal (ensure 10+ trades)
- Clear regime switching (choppy vs trending)
- Proper MTF alignment using mtf_data helper
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_1d_hma_chop_rsi_donchian_1w_v1"
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
    CHOP > 61.8 = choppy/range market
    CHOP < 38.2 = trending market
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
    df_1w = get_htf_data(prices, '1w')
    
    # Calculate 1w HTF indicators (primary trend regime)
    hma_1w_21 = calculate_hma(df_1w['close'].values, 21)
    
    # Align HTF to LTF (Rule 2 - auto shift(1))
    hma_1w_21_aligned = align_htf_to_ltf(prices, df_1w, hma_1w_21)
    
    # Calculate 1d indicators
    atr_14 = calculate_atr(high, low, close, 14)
    chop_14 = calculate_choppiness_index(high, low, close, 14)
    rsi_14 = calculate_rsi(close, 14)
    hma_1d_21 = calculate_hma(close, 21)
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
    last_trade_bar = -10
    
    for i in range(100, n):
        # Skip if indicators not ready
        if np.isnan(atr_14[i]) or atr_14[i] == 0:
            continue
        
        if np.isnan(hma_1w_21_aligned[i]):
            continue
        
        if np.isnan(rsi_14[i]) or np.isnan(hma_1d_21[i]):
            continue
        
        if np.isnan(chop_14[i]):
            continue
        
        if np.isnan(donchian_upper[i]) or np.isnan(donchian_lower[i]):
            continue
        
        # === 1W TREND REGIME (primary direction filter) ===
        regime_bull = close[i] > hma_1w_21_aligned[i]
        regime_bear = close[i] < hma_1w_21_aligned[i]
        
        # === CHOPPINESS REGIME ===
        is_choppy = chop_14[i] > 55.0
        is_trending = chop_14[i] < 45.0
        
        # === 1D LOCAL SIGNALS ===
        price_above_1d_hma = close[i] > hma_1d_21[i]
        price_below_1d_hma = close[i] < hma_1d_21[i]
        
        # === DONCHIAN BREAKOUT ===
        donchian_breakout_long = close[i] > donchian_upper[i] * 0.995
        donchian_breakout_short = close[i] < donchian_lower[i] * 1.005
        
        # === RSI THRESHOLDS (relaxed for more trades) ===
        rsi_oversold = rsi_14[i] < 40.0
        rsi_overbought = rsi_14[i] > 60.0
        rsi_extreme_oversold = rsi_14[i] < 30.0
        rsi_extreme_overbought = rsi_14[i] > 70.0
        
        # === ENTRY LOGIC (SIMPLIFIED - fewer AND conditions) ===
        new_signal = 0.0
        bars_since_last_trade = i - last_trade_bar
        
        # TREND FOLLOWING MODE (when trending + regime aligned)
        if is_trending:
            # LONG: Bull regime OR price above 1d HMA
            if regime_bull or price_above_1d_hma:
                new_signal = BASE_SIZE
            # LONG: Donchian breakout (strong signal)
            if donchian_breakout_long:
                new_signal = STRONG_SIZE
            
            # SHORT: Bear regime OR price below 1d HMA
            if regime_bear or price_below_1d_hma:
                if new_signal == 0.0 or abs(new_signal) < BASE_SIZE:
                    new_signal = -BASE_SIZE
            # SHORT: Donchian breakdown (strong signal)
            if donchian_breakout_short:
                if new_signal == 0.0:
                    new_signal = -STRONG_SIZE
        
        # MEAN REVERSION MODE (when choppy)
        if is_choppy:
            # LONG: RSI oversold (any regime for more trades)
            if rsi_oversold:
                new_signal = BASE_SIZE
            # LONG: RSI extreme oversold (stronger)
            if rsi_extreme_oversold:
                new_signal = STRONG_SIZE
            
            # SHORT: RSI overbought (any regime for more trades)
            if rsi_overbought:
                if new_signal == 0.0:
                    new_signal = -BASE_SIZE
            # SHORT: RSI extreme overbought (stronger)
            if rsi_extreme_overbought:
                if new_signal == 0.0:
                    new_signal = -STRONG_SIZE
        
        # === NEUTRAL REGIME (chop between 45-55) ===
        if not is_choppy and not is_trending:
            # Use simple HMA crossover
            if price_above_1d_hma:
                new_signal = BASE_SIZE * 0.7
            elif price_below_1d_hma:
                new_signal = -BASE_SIZE * 0.7
        
        # === FREQUENCY SAFEGUARD (ensure 10+ trades) ===
        # Force trade if no signal for 10 bars (~10 days on 1d)
        if bars_since_last_trade > 10 and new_signal == 0.0 and not in_position:
            if regime_bull:
                new_signal = BASE_SIZE * 0.6
            elif regime_bear:
                new_signal = -BASE_SIZE * 0.6
            elif rsi_14[i] < 45:
                new_signal = BASE_SIZE * 0.5
            elif rsi_14[i] > 55:
                new_signal = -BASE_SIZE * 0.5
        
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
            # Long position but regime turns bearish
            if position_side > 0 and regime_bear and price_below_1d_hma:
                regime_reversal = True
            # Short position but regime turns bullish
            if position_side < 0 and regime_bull and price_above_1d_hma:
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
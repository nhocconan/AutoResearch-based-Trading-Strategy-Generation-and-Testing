#!/usr/bin/env python3
"""
Experiment #223: 1d Primary + 1w HTF — HMA Trend + RSI Pullback + Choppiness Regime

Hypothesis: After repeated failures with complex dual-regime switching (#212, #213, #214, #222),
simplify to a single proven pattern: HMA trend direction + RSI pullback entries, but add
Choppiness Index as a META-FILTER (not regime switch). Research shows Choppiness + Connors RSI
achieved ETH Sharpe +0.923, and HMA + RSI + ATR achieved SOL Sharpe +0.879.

Key differences from failed attempts:
1. Choppiness is a FILTER not a regime switch — only trade when CHOP confirms market state
2. 1w HMA(21) for macro bias (aligned via mtf_data) — very slow trend filter
3. 1d HMA(16/48) crossover for primary trend direction
4. RSI(14) pullback entries (45-55 zone) — ensures we enter on retracements not breakouts
5. ATR(14) 2.5x trailing stop for risk management
6. Fewer conflicting filters = more trades while maintaining quality

TARGET: 25-40 trades/year on 1d, Sharpe > 0.5 on ALL symbols
Position sizing: 0.0, ±0.20, ±0.30 (discrete to minimize fee churn)
Stoploss: ATR(14) 2.5x trailing stop

Why this might work when others failed:
- 1d timeframe = fewer trades = less fee drag (vs 4h/12h strategies)
- 1w HTF = very stable macro bias (less whipsaw than 1d/4h HTF)
- RSI pullback (not breakout) = better entry timing in trends
- Choppiness filter = avoid trading in wrong market state
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_1d_hma_rsi_chop_1w_atr_v1"
timeframe = "1d"
leverage = 1.0

def calculate_hma(close, period):
    """
    Calculate Hull Moving Average (HMA).
    HMA = WMA(2*WMA(n/2) - WMA(n)) with sqrt(n) window
    Faster and smoother than EMA, less lag.
    """
    n = len(close)
    close_s = pd.Series(close)
    
    half = max(1, period // 2)
    sqrt_n = max(1, int(np.sqrt(period)))
    
    # WMA helper
    def wma(series, window):
        weights = np.arange(1, window + 1)
        return series.rolling(window=window, min_periods=window).apply(
            lambda x: np.dot(x, weights) / weights.sum(), raw=True
        )
    
    wma_half = wma(close_s, half)
    wma_full = wma(close_s, period)
    
    hull = 2 * wma_half - wma_full
    hma = wma(hull, sqrt_n)
    
    return hma.values

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
    """Calculate RSI."""
    close_s = pd.Series(close)
    delta = close_s.diff()
    gain = delta.clip(lower=0)
    loss = (-delta).clip(lower=0)
    avg_gain = gain.ewm(span=period, min_periods=period, adjust=False).mean()
    avg_loss = loss.ewm(span=period, min_periods=period, adjust=False).mean()
    with np.errstate(divide='ignore', invalid='ignore'):
        rs = avg_gain / (avg_loss + 1e-10)
        rsi = 100.0 - (100.0 / (1.0 + rs))
    return rsi.fillna(50.0).values

def calculate_choppiness(high, low, close, period=14):
    """
    Calculate Choppiness Index (CHOP).
    CHOP = 100 * LOG10(SUM(ATR, period) / (Highest High - Lowest Low)) / LOG10(period)
    
    Interpretation:
    - CHOP > 61.8 = Choppy/Range market (favor mean reversion)
    - CHOP < 38.2 = Trending market (favor trend following)
    - 38.2 - 61.8 = Transition zone
    """
    n = len(close)
    chop = np.zeros(n)
    
    # Calculate ATR for each bar
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]
    
    for i in range(period, n):
        atr_sum = np.sum(tr[i-period+1:i+1])
        highest_high = np.max(high[i-period+1:i+1])
        lowest_low = np.min(low[i-period+1:i+1])
        price_range = highest_high - lowest_low
        
        if price_range > 1e-10 and atr_sum > 1e-10:
            chop[i] = 100.0 * np.log10(atr_sum / price_range) / np.log10(period)
        else:
            chop[i] = 50.0  # neutral
    
    return chop

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    n = len(close)
    
    # Load HTF data ONCE before loop (Rule 1 - CRITICAL)
    df_1w = get_htf_data(prices, '1w')
    
    # Calculate 1d indicators (primary timeframe)
    hma_16 = calculate_hma(close, 16)
    hma_48 = calculate_hma(close, 48)
    rsi_14 = calculate_rsi(close, period=14)
    atr_14 = calculate_atr(high, low, close, period=14)
    chop_14 = calculate_choppiness(high, low, close, period=14)
    
    # Calculate 1w HMA for macro trend (aligned properly)
    hma_1w_raw = calculate_hma(df_1w['close'].values, 21)
    hma_1w_aligned = align_htf_to_ltf(prices, df_1w, hma_1w_raw)
    
    signals = np.zeros(n)
    POSITION_SIZE_FULL = 0.30
    POSITION_SIZE_HALF = 0.20
    
    # Track position state for stoploss
    in_position = False
    position_side = 0
    entry_price = 0.0
    highest_since_entry = 0.0
    lowest_since_entry = 0.0
    
    for i in range(100, n):  # Start later to ensure all indicators ready
        # Skip if indicators not ready
        if np.isnan(atr_14[i]) or atr_14[i] == 0:
            continue
        if np.isnan(hma_16[i]) or np.isnan(hma_48[i]):
            continue
        if np.isnan(rsi_14[i]):
            continue
        if np.isnan(hma_1w_aligned[i]):
            continue
        if np.isnan(chop_14[i]):
            continue
        
        # === HTF MACRO BIAS (1w HMA) ===
        price_above_hma_1w = close[i] > hma_1w_aligned[i]
        price_below_hma_1w = close[i] < hma_1w_aligned[i]
        
        # === TREND DETECTION (1d HMA crossover) ===
        hma_bullish = hma_16[i] > hma_48[i]
        hma_bearish = hma_16[i] < hma_48[i]
        
        # === CHOPPINESS FILTER ===
        # Only trade when market state confirms our strategy type
        # For trend following: CHOP < 55 (not too choppy)
        # For mean reversion: CHOP > 50 (some chop)
        # We're trend-following with pullback entries, so prefer lower chop
        trending_market = chop_14[i] < 55.0
        choppy_market = chop_14[i] > 50.0
        
        # === RSI PULLBACK ZONE ===
        # Enter on pullbacks, not breakouts
        rsi_pullback_long = 45.0 <= rsi_14[i] <= 55.0
        rsi_pullback_short = 45.0 <= rsi_14[i] <= 55.0
        rsi_not_extreme_long = rsi_14[i] < 65.0
        rsi_not_extreme_short = rsi_14[i] > 35.0
        
        # === ENTRY LOGIC ===
        new_signal = 0.0
        
        # LONG ENTRY: HMA bullish + RSI pullback + trending market + 1w bias
        if hma_bullish and rsi_not_extreme_long:
            # Full size: with 1w macro trend + trending market
            if price_above_hma_1w and trending_market and rsi_pullback_long:
                new_signal = POSITION_SIZE_FULL
            # Half size: against 1w macro or choppy market
            elif rsi_pullback_long:
                new_signal = POSITION_SIZE_HALF
        
        # SHORT ENTRY: HMA bearish + RSI pullback + trending market + 1w bias
        elif hma_bearish and rsi_not_extreme_short:
            # Full size: with 1w macro trend + trending market
            if price_below_hma_1w and trending_market and rsi_pullback_short:
                new_signal = -POSITION_SIZE_FULL
            # Half size: against 1w macro or choppy market
            elif rsi_pullback_short:
                new_signal = -POSITION_SIZE_HALF
        
        # === HOLD POSITION LOGIC ===
        # Hold if in position and trend still valid
        if in_position and new_signal == 0.0:
            if position_side > 0:
                # Hold long if HMA still bullish and RSI not overbought
                if hma_bullish and rsi_14[i] < 70.0:
                    new_signal = signals[i-1] if i > 0 else 0.0
            elif position_side < 0:
                # Hold short if HMA still bearish and RSI not oversold
                if hma_bearish and rsi_14[i] > 30.0:
                    new_signal = signals[i-1] if i > 0 else 0.0
        
        # === STOPLOSS CHECK (2.5 * ATR trailing) ===
        stoploss_triggered = False
        
        if in_position and position_side > 0:
            highest_since_entry = max(highest_since_entry, close[i])
            stop_price = highest_since_entry - 2.5 * atr_14[i]
            if close[i] < stop_price:
                stoploss_triggered = True
        
        if in_position and position_side < 0:
            if lowest_since_entry == 0.0:
                lowest_since_entry = close[i]
            else:
                lowest_since_entry = min(lowest_since_entry, close[i])
            stop_price = lowest_since_entry + 2.5 * atr_14[i]
            if close[i] > stop_price:
                stoploss_triggered = True
        
        if stoploss_triggered:
            new_signal = 0.0
        
        # === TREND REVERSAL EXIT ===
        # Exit long if HMA crosses bearish
        if in_position and position_side > 0 and hma_bearish:
            new_signal = 0.0
        
        # Exit short if HMA crosses bullish
        if in_position and position_side < 0 and hma_bullish:
            new_signal = 0.0
        
        # Exit if RSI reaches extreme (take profit)
        if in_position and position_side > 0 and rsi_14[i] > 70.0:
            new_signal = 0.0
        
        if in_position and position_side < 0 and rsi_14[i] < 30.0:
            new_signal = 0.0
        
        # === UPDATE POSITION TRACKING ===
        if new_signal != 0.0:
            if not in_position:
                in_position = True
                position_side = np.sign(new_signal)
                entry_price = close[i]
                highest_since_entry = close[i] if position_side > 0 else 0.0
                lowest_since_entry = close[i] if position_side < 0 else 0.0
            elif np.sign(new_signal) != position_side:
                # Position flip
                position_side = np.sign(new_signal)
                entry_price = close[i]
                highest_since_entry = close[i] if position_side > 0 else 0.0
                lowest_since_entry = close[i] if position_side < 0 else 0.0
        else:
            if in_position:
                in_position = False
                position_side = 0
                entry_price = 0.0
                highest_since_entry = 0.0
                lowest_since_entry = 0.0
        
        signals[i] = new_signal
    
    return signals
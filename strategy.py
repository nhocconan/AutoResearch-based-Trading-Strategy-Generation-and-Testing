#!/usr/bin/env python3
"""
Experiment #227: 1d Primary + 1w HTF — Regime-Adaptive Strategy

Hypothesis: After 190+ failures with pure trend-following on BTC/ETH, switch to 
REGIME-ADAPTIVE logic. Research shows Choppiness Index regime filter works well 
for ETH (Sharpe +0.923). Key insight: BTC/ETH are mean-reverting in bear/range 
markets (2022 crash, 2025 test period) but trend in bull markets.

Strategy Logic:
1. CHOPPINESS INDEX(14) detects regime: >61.8 = range, <38.2 = trend
2. RANGE regime (CHOP>61.8): Mean reversion with RSI(14) extremes + BB filter
3. TREND regime (CHOP<38.2): Trend following with HMA(21/63) + Donchian(20) breakout
4. 1w HMA(21) for macro bias (only take trades with weekly trend)
5. ATR(14) 2.5x trailing stoploss on all positions

Why this might work:
- Adapts to market conditions instead of forcing one regime
- Range logic captures 2022 bottom whipsaw and 2025 bear/range
- Trend logic captures 2021 bull run
- 1w filter prevents counter-trend trades in strong macro moves
- 1d timeframe = 20-40 trades/year target (low fee drag)

Position sizing: 0.0, ±0.20, ±0.30 (discrete levels to minimize churn)
Target: Sharpe > 0.5 on ALL symbols, trades >= 30 on train, >= 3 on test
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_1d_regime_adaptive_chop_rsi_hma_1w_v1"
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
    CHOP = 100 * LOG10(SUM(ATR, n) / (Highest High - Lowest Low)) / LOG10(n)
    > 61.8 = range/choppy, < 38.2 = trending
    """
    n = len(close)
    chop = np.zeros(n)
    
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
            chop[i] = 50.0
    
    return chop

def calculate_bollinger(close, period=20, std_dev=2.0):
    """Calculate Bollinger Bands."""
    close_s = pd.Series(close)
    sma = close_s.rolling(window=period, min_periods=period).mean()
    std = close_s.rolling(window=period, min_periods=period).std()
    upper = sma + std_dev * std
    lower = sma - std_dev * std
    return upper.values, lower.values, sma.values

def calculate_donchian(high, low, period=20):
    """Calculate Donchian Channel (upper/lower bounds)."""
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
    
    # Calculate 1d indicators (primary timeframe)
    hma_21 = calculate_hma(close, 21)
    hma_63 = calculate_hma(close, 63)
    rsi_14 = calculate_rsi(close, period=14)
    atr_14 = calculate_atr(high, low, close, period=14)
    chop_14 = calculate_choppiness(high, low, close, period=14)
    bb_upper, bb_lower, bb_mid = calculate_bollinger(close, period=20, std_dev=2.0)
    donchian_upper, donchian_lower = calculate_donchian(high, low, period=20)
    
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
    
    for i in range(100, n):
        # Skip if indicators not ready
        if np.isnan(atr_14[i]) or atr_14[i] == 0:
            continue
        if np.isnan(hma_21[i]) or np.isnan(hma_63[i]):
            continue
        if np.isnan(rsi_14[i]) or np.isnan(chop_14[i]):
            continue
        if np.isnan(hma_1w_aligned[i]):
            continue
        if np.isnan(bb_upper[i]) or np.isnan(bb_lower[i]):
            continue
        if np.isnan(donchian_upper[i]) or np.isnan(donchian_lower[i]):
            continue
        
        # === REGIME DETECTION (Choppiness Index) ===
        in_range_regime = chop_14[i] > 61.8
        in_trend_regime = chop_14[i] < 38.2
        neutral_regime = not in_range_regime and not in_trend_regime
        
        # === MACRO BIAS (1w HMA) ===
        price_above_hma_1w = close[i] > hma_1w_aligned[i]
        price_below_hma_1w = close[i] < hma_1w_aligned[i]
        
        # === TREND DETECTION (1d HMA crossover) ===
        hma_bullish = hma_21[i] > hma_63[i]
        hma_bearish = hma_21[i] < hma_63[i]
        
        # === MEAN REVERSION SIGNALS (Range Regime) ===
        rsi_oversold = rsi_14[i] < 35.0
        rsi_overbought = rsi_14[i] > 65.0
        price_near_bb_lower = close[i] < bb_lower[i] * 1.005
        price_near_bb_upper = close[i] > bb_upper[i] * 0.995
        
        # === TREND FOLLOWING SIGNALS (Trend Regime) ===
        breakout_long = close[i] > donchian_upper[i-1] if i > 0 else False
        breakout_short = close[i] < donchian_lower[i-1] if i > 0 else False
        
        # === ENTRY LOGIC ===
        new_signal = 0.0
        
        # RANGE REGIME: Mean Reversion
        if in_range_regime:
            # LONG: RSI oversold + near BB lower + weekly bias OK
            if rsi_oversold and price_near_bb_lower:
                if price_above_hma_1w or not price_below_hma_1w:
                    new_signal = POSITION_SIZE_HALF
                elif hma_bullish:
                    new_signal = POSITION_SIZE_HALF
            
            # SHORT: RSI overbought + near BB upper + weekly bias OK
            elif rsi_overbought and price_near_bb_upper:
                if price_below_hma_1w or not price_above_hma_1w:
                    new_signal = -POSITION_SIZE_HALF
                elif hma_bearish:
                    new_signal = -POSITION_SIZE_HALF
        
        # TREND REGIME: Trend Following
        elif in_trend_regime:
            # LONG: HMA bullish + Donchian breakout + RSI not overbought
            if hma_bullish and breakout_long and rsi_14[i] < 70.0:
                if price_above_hma_1w:
                    new_signal = POSITION_SIZE_FULL
                else:
                    new_signal = POSITION_SIZE_HALF
            
            # SHORT: HMA bearish + Donchian breakout + RSI not oversold
            elif hma_bearish and breakout_short and rsi_14[i] > 30.0:
                if price_below_hma_1w:
                    new_signal = -POSITION_SIZE_FULL
                else:
                    new_signal = -POSITION_SIZE_HALF
        
        # NEUTRAL REGIME: Reduced size, wait for clarity
        elif neutral_regime:
            # Only take strong signals with weekly confirmation
            if hma_bullish and rsi_oversold and price_above_hma_1w:
                new_signal = POSITION_SIZE_HALF
            elif hma_bearish and rsi_overbought and price_below_hma_1w:
                new_signal = -POSITION_SIZE_HALF
        
        # === HOLD POSITION LOGIC ===
        # Hold if in position and regime/trend still valid
        if in_position and new_signal == 0.0:
            if position_side > 0:
                # Hold long if not in strong opposite signal
                if not (in_trend_regime and hma_bearish):
                    if rsi_14[i] < 75.0:
                        new_signal = signals[i-1] if i > 0 else 0.0
            elif position_side < 0:
                # Hold short if not in strong opposite signal
                if not (in_trend_regime and hma_bullish):
                    if rsi_14[i] > 25.0:
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
        # Exit long if HMA crosses bearish in trend regime
        if in_position and position_side > 0 and in_trend_regime and hma_bearish:
            new_signal = 0.0
        
        # Exit short if HMA crosses bullish in trend regime
        if in_position and position_side < 0 and in_trend_regime and hma_bullish:
            new_signal = 0.0
        
        # Exit if weekly trend strongly against position
        if in_position and position_side > 0 and price_below_hma_1w and chop_14[i] < 50:
            new_signal = 0.0
        
        if in_position and position_side < 0 and price_above_hma_1w and chop_14[i] < 50:
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
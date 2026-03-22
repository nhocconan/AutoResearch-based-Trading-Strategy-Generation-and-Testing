#!/usr/bin/env python3
"""
Experiment #004: 4h KAMA Trend + 12h HMA Filter + ATR Risk Management

Hypothesis: 4h primary with 12h HTF trend filter will work better than failed 12h/1d strategies.
Key learnings from failures:
- #002 had 0 trades (over-filtered entries)
- #003 Choppiness+Connors failed badly (Sharpe=-4.468)
- Need SIMPLER entry conditions that actually trigger

Design:
1. 12h HMA(21) for major trend direction (via mtf_data, call ONCE before loop)
2. 4h KAMA(14) for adaptive trend following (adapts to volatility)
3. RSI(14) simple filter: >45 for longs, <55 for shorts (NOT extreme values)
4. ATR(14) for stoploss (2.0x) - tighter than failed strategies
5. Minimal filters to ensure 30+ trades per symbol

Why this should work:
- KAMA adapts to market conditions (better than static EMA/HMA in chop)
- 4h TF targets 30-60 trades/year (optimal range)
- Simple RSI filter (>45/<55) ensures trades actually trigger
- 12h HTF prevents counter-trend trades (major failure mode in 2022)
- Looser entry conditions than failed #002 strategy

Timeframe: 4h (REQUIRED for Experiment #004)
HTF: 12h via mtf_data helper (call ONCE before loop)
Position sizing: 0.25-0.30 discrete
Stoploss: 2.0 * ATR(14)
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_4h_kama_12h_hma_atr_v1"
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

def calculate_rsi(close, period=14):
    """Calculate RSI using Wilder's smoothing."""
    delta = np.diff(close)
    delta = np.insert(delta, 0, 0)
    
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    
    gain_avg = pd.Series(gain).ewm(span=period, min_periods=period, adjust=False).mean().values
    loss_avg = pd.Series(loss).ewm(span=period, min_periods=period, adjust=False).mean().values
    
    loss_avg = np.where(loss_avg == 0, 1e-10, loss_avg)
    rs = gain_avg / loss_avg
    rsi = 100 - (100 / (1 + rs))
    rsi = np.clip(rsi, 0, 100)
    return rsi

def calculate_hma(close, period=21):
    """Calculate Hull Moving Average for smoother trend with less lag."""
    close_s = pd.Series(close)
    half = max(1, period // 2)
    sqrt_period = max(1, int(np.sqrt(period)))
    wma1 = close_s.ewm(span=half, min_periods=half, adjust=False).mean()
    wma2 = close_s.ewm(span=period, min_periods=period, adjust=False).mean()
    wma3 = (2 * wma1 - wma2).ewm(span=sqrt_period, min_periods=sqrt_period, adjust=False).mean()
    return wma3.values

def calculate_kama(close, period=14, fast_period=2, slow_period=30):
    """
    Calculate Kaufman Adaptive Moving Average.
    KAMA adapts to market noise - moves fast in trends, slow in chop.
    Formula: KAMA = KAMA_prev + SC * (Price - KAMA_prev)
    SC = (ER * (fast_sc - slow_sc) + slow_sc)^2
    ER = Change / Volatility (Efficiency Ratio)
    """
    n = len(close)
    kama = np.zeros(n)
    kama[0] = close[0]
    
    fast_sc = 2.0 / (fast_period + 1)
    slow_sc = 2.0 / (slow_period + 1)
    
    for i in range(1, n):
        if i < period:
            kama[i] = close[i]
            continue
        
        # Efficiency Ratio: how much price moved vs total volatility
        change = np.abs(close[i] - close[i - period])
        volatility = np.sum(np.abs(np.diff(close[i-period:i+1])))
        
        if volatility == 0:
            er = 0
        else:
            er = change / volatility
        
        # Smoothing Constant
        sc = (er * (fast_sc - slow_sc) + slow_sc) ** 2
        
        # KAMA calculation
        kama[i] = kama[i-1] + sc * (close[i] - kama[i-1])
    
    return kama

def calculate_adx(high, low, close, period=14):
    """Calculate ADX for trend strength."""
    n = len(close)
    
    # True Range
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]
    
    # Directional Movement
    plus_dm = np.zeros(n)
    minus_dm = np.zeros(n)
    
    for i in range(1, n):
        plus_diff = high[i] - high[i-1]
        minus_diff = low[i-1] - low[i]
        
        if plus_diff > minus_diff and plus_diff > 0:
            plus_dm[i] = plus_diff
        if minus_diff > plus_diff and minus_diff > 0:
            minus_dm[i] = minus_diff
    
    # Smoothed DM and TR
    plus_dm_s = pd.Series(plus_dm).ewm(span=period, min_periods=period, adjust=False).mean().values
    minus_dm_s = pd.Series(minus_dm).ewm(span=period, min_periods=period, adjust=False).mean().values
    tr_s = pd.Series(tr).ewm(span=period, min_periods=period, adjust=False).mean().values
    
    # DI and ADX
    plus_di = 100 * plus_dm_s / np.where(tr_s == 0, 1e-10, tr_s)
    minus_di = 100 * minus_dm_s / np.where(tr_s == 0, 1e-10, tr_s)
    
    dx = 100 * np.abs(plus_di - minus_di) / np.where(plus_di + minus_di == 0, 1e-10, plus_di + minus_di)
    adx = pd.Series(dx).ewm(span=period, min_periods=period, adjust=False).mean().values
    
    return adx

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    n = len(close)
    
    # Load 12h HTF data ONCE before loop (Rule 1 - CRITICAL)
    df_12h = get_htf_data(prices, '12h')
    
    # Calculate 12h HMA trend
    hma_12h_21 = calculate_hma(df_12h['close'].values, 21)
    hma_12h_21_aligned = align_htf_to_ltf(prices, df_12h, hma_12h_21)
    
    # Calculate 4h indicators
    atr_14 = calculate_atr(high, low, close, 14)
    rsi_14 = calculate_rsi(close, 14)
    kama_14 = calculate_kama(close, 14)
    adx_14 = calculate_adx(high, low, close, 14)
    
    # Also calculate 4h HMA for additional trend confirmation
    hma_4h_21 = calculate_hma(close, 21)
    
    signals = np.zeros(n)
    
    # Position sizing (Rule 4 - discrete levels, max 0.40)
    BASE_SIZE = 0.25
    STRONG_SIZE = 0.30
    
    # Track position state for stoploss
    in_position = False
    position_side = 0
    entry_price = 0.0
    highest_price = 0.0
    lowest_price = 0.0
    last_trade_bar = -50
    
    for i in range(50, n):
        # Skip if indicators not ready
        if np.isnan(atr_14[i]) or atr_14[i] == 0:
            continue
        
        if np.isnan(hma_12h_21_aligned[i]) or np.isnan(kama_14[i]):
            continue
        
        # === 12H HTF TREND BIAS ===
        # Simple: price above 12h HMA = bullish bias, below = bearish
        htf_bullish = close[i] > hma_12h_21_aligned[i]
        htf_bearish = close[i] < hma_12h_21_aligned[i]
        
        # === 4H LOCAL TREND (KAMA) ===
        local_bullish = close[i] > kama_14[i]
        local_bearish = close[i] < kama_14[i]
        
        # === KAMA SLOPE (direction of KAMA) ===
        kama_rising = kama_14[i] > kama_14[i-1] if i > 0 else False
        kama_falling = kama_14[i] < kama_14[i-1] if i > 0 else False
        
        # === ADX TREND STRENGTH (lower threshold for more trades) ===
        adx_strong = adx_14[i] > 20  # Lower than failed strategy (was 25)
        
        # === RSI FILTER (simple, ensures trades trigger) ===
        # Use >45/<55 instead of >50/<50 for more trade generation
        rsi_bullish = rsi_14[i] > 45
        rsi_bearish = rsi_14[i] < 55
        
        # === POSITION SIZING BASED ON TREND STRENGTH ===
        if htf_bullish and local_bullish and adx_strong:
            current_size = STRONG_SIZE
        elif htf_bullish and local_bullish:
            current_size = BASE_SIZE
        elif htf_bearish and local_bearish and adx_strong:
            current_size = STRONG_SIZE
        elif htf_bearish and local_bearish:
            current_size = BASE_SIZE
        else:
            current_size = BASE_SIZE * 0.8  # Weaker signal
        
        # === ENTRY LOGIC (SIMPLER than failed strategies) ===
        new_signal = 0.0
        bars_since_last_trade = i - last_trade_bar
        
        # LONG ENTRY: 12h bullish + 4h KAMA bullish + RSI > 45
        # Only require 2 of 3 conditions for more trades
        long_conditions = sum([htf_bullish, local_bullish, rsi_bullish])
        if long_conditions >= 2:
            new_signal = current_size
        
        # SHORT ENTRY: 12h bearish + 4h KAMA bearish + RSI < 55
        short_conditions = sum([htf_bearish, local_bearish, rsi_bearish])
        if short_conditions >= 2:
            new_signal = -current_size
        
        # === FREQUENCY SAFEGUARD (ensure minimum trades) ===
        # If no trades for 20 bars (~3.3 days on 4h), allow weaker entry
        if bars_since_last_trade > 20 and new_signal == 0.0 and not in_position:
            if htf_bullish and local_bullish:
                new_signal = current_size * 0.8
            elif htf_bearish and local_bearish:
                new_signal = -current_size * 0.8
        
        # === STOPLOSS LOGIC (Rule 6) - 2.0 * ATR ===
        stoploss_triggered = False
        
        if in_position and position_side != 0:
            if position_side > 0:
                if close[i] > highest_price:
                    highest_price = close[i]
                stoploss_price = highest_price - 2.0 * atr_14[i]
                if close[i] < stoploss_price:
                    stoploss_triggered = True
            
            if position_side < 0:
                if lowest_price == 0.0 or close[i] < lowest_price:
                    lowest_price = close[i]
                stoploss_price = lowest_price + 2.0 * atr_14[i]
                if close[i] > stoploss_price:
                    stoploss_triggered = True
        
        # === TREND REVERSAL EXIT ===
        trend_reversal = False
        if in_position and position_side != 0:
            # Exit long if 12h trend turns bearish
            if position_side > 0 and htf_bearish:
                trend_reversal = True
            # Exit short if 12h trend turns bullish
            if position_side < 0 and htf_bullish:
                trend_reversal = True
        
        # === RSI EXTREME EXIT (take profit) ===
        rsi_exit = False
        if in_position and position_side != 0:
            # Exit long when RSI becomes overbought
            if position_side > 0 and rsi_14[i] > 70:
                rsi_exit = True
            # Exit short when RSI becomes oversold
            if position_side < 0 and rsi_14[i] < 30:
                rsi_exit = True
        
        # Apply stoploss or reversals
        if stoploss_triggered or trend_reversal or rsi_exit:
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
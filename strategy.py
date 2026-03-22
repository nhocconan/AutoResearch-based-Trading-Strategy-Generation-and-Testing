#!/usr/bin/env python3
"""
Experiment #002: 12h HMA Trend + RSI Pullback + 1d HTF Filter + ATR Stop

Hypothesis: 12h primary with 1d HTF trend filter captures major moves while RSI 
pullback entries avoid chasing. Key improvements over failed #001:
1. Simpler entry logic (RSI extremes vs Donchian breakout)
2. Allow entries without perfect HTF alignment after cooldown period
3. HMA crossover for exit timing (proven on SOL Sharpe +0.879)
4. More generous RSI thresholds to ensure trade generation

Key design:
1. 1d HMA(21) for major trend direction (via mtf_data, call ONCE)
2. 12h HMA(16/48) for local trend and exit signals
3. RSI(14) pullback: <40 for longs, >60 for shorts (generous thresholds)
4. ADX(14) > 18 for trend confirmation (lower threshold = more trades)
5. ATR(14) 2.5x trailing stoploss
6. Discrete sizing: 0.25 base, 0.30 strong trend

Why this should work:
- HMA proven on SOL (Sharpe +0.879) with less lag than EMA
- RSI pullback ensures we enter on dips, not tops
- 1d HTF filter prevents counter-trend (2022 crash protection)
- Lower ADX threshold (18 vs 25) = more trade opportunities
- Frequency safeguard ensures minimum trades generated

Timeframe: 12h (REQUIRED)
HTF: 1d via mtf_data helper (call ONCE before loop)
Position sizing: 0.20-0.30 discrete
Stoploss: 2.5 * ATR(14)
Target trades: 25-50/year (12h TF optimal range)
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_12h_hma_rsi_1d_filter_atr_v2"
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

def calculate_adx(high, low, close, period=14):
    """Calculate ADX for trend strength."""
    n = len(close)
    
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]
    
    plus_dm = np.zeros(n)
    minus_dm = np.zeros(n)
    
    for i in range(1, n):
        plus_diff = high[i] - high[i-1]
        minus_diff = low[i-1] - low[i]
        
        if plus_diff > minus_diff and plus_diff > 0:
            plus_dm[i] = plus_diff
        if minus_diff > plus_diff and minus_diff > 0:
            minus_dm[i] = minus_diff
    
    plus_dm_s = pd.Series(plus_dm).ewm(span=period, min_periods=period, adjust=False).mean().values
    minus_dm_s = pd.Series(minus_dm).ewm(span=period, min_periods=period, adjust=False).mean().values
    tr_s = pd.Series(tr).ewm(span=period, min_periods=period, adjust=False).mean().values
    
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
    
    # Load 1d HTF data ONCE before loop (Rule 1 - CRITICAL)
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate 1d HMA trend
    hma_1d_21 = calculate_hma(df_1d['close'].values, 21)
    hma_1d_21_aligned = align_htf_to_ltf(prices, df_1d, hma_1d_21)
    
    # Calculate 12h indicators
    atr_14 = calculate_atr(high, low, close, 14)
    rsi_14 = calculate_rsi(close, 14)
    adx_14 = calculate_adx(high, low, close, 14)
    
    # 12h HMA for local trend (fast/slow crossover)
    hma_12h_fast = calculate_hma(close, 16)
    hma_12h_slow = calculate_hma(close, 48)
    
    signals = np.zeros(n)
    
    # Position sizing (Rule 4 - discrete levels, max 0.40)
    BASE_SIZE = 0.25
    STRONG_SIZE = 0.30
    WEAK_SIZE = 0.20
    
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
        
        if np.isnan(hma_1d_21_aligned[i]) or np.isnan(hma_12h_fast[i]) or np.isnan(hma_12h_slow[i]):
            continue
        
        if np.isnan(rsi_14[i]) or np.isnan(adx_14[i]):
            continue
        
        # === 1D HTF TREND BIAS ===
        htf_bullish = close[i] > hma_1d_21_aligned[i]
        htf_bearish = close[i] < hma_1d_21_aligned[i]
        
        # === 12H LOCAL TREND ===
        local_bullish = hma_12h_fast[i] > hma_12h_slow[i]
        local_bearish = hma_12h_fast[i] < hma_12h_slow[i]
        
        # HMA crossover signals (for entry timing)
        hma_cross_long = hma_12h_fast[i] > hma_12h_slow[i] and hma_12h_fast[i-1] <= hma_12h_slow[i-1]
        hma_cross_short = hma_12h_fast[i] < hma_12h_slow[i] and hma_12h_fast[i-1] >= hma_12h_slow[i-1]
        
        # === ADX TREND STRENGTH (lower threshold = more trades) ===
        adx_strong = adx_14[i] > 18
        
        # === RSI PULLBACK FILTER (generous thresholds for trade generation) ===
        rsi_oversold = rsi_14[i] < 45  # Long entry zone
        rsi_overbought = rsi_14[i] > 55  # Short entry zone
        rsi_extreme_long = rsi_14[i] < 35  # Strong long signal
        rsi_extreme_short = rsi_14[i] > 65  # Strong short signal
        
        # === POSITION SIZING ===
        if htf_bullish and local_bullish and adx_strong:
            current_size = STRONG_SIZE
        elif htf_bullish and local_bullish:
            current_size = BASE_SIZE
        elif htf_bullish:
            current_size = WEAK_SIZE
        elif htf_bearish and local_bearish and adx_strong:
            current_size = STRONG_SIZE
        elif htf_bearish and local_bearish:
            current_size = BASE_SIZE
        elif htf_bearish:
            current_size = WEAK_SIZE
        else:
            current_size = BASE_SIZE
        
        # === ENTRY LOGIC ===
        new_signal = 0.0
        bars_since_last_trade = i - last_trade_bar
        
        # LONG ENTRY: Multiple conditions (any can trigger)
        # Primary: HTF bullish + RSI pullback + local bullish
        if htf_bullish and rsi_oversold and local_bullish:
            new_signal = current_size
        # Secondary: HMA crossover + RSI not overbought
        elif hma_cross_long and rsi_14[i] < 60:
            new_signal = current_size
        # Tertiary: Strong RSI extreme + HTF bias
        elif htf_bullish and rsi_extreme_long:
            new_signal = current_size
        
        # SHORT ENTRY: Multiple conditions (any can trigger)
        # Primary: HTF bearish + RSI rally + local bearish
        elif htf_bearish and rsi_overbought and local_bearish:
            new_signal = -current_size
        # Secondary: HMA crossover + RSI not oversold
        elif hma_cross_short and rsi_14[i] > 40:
            new_signal = -current_size
        # Tertiary: Strong RSI extreme + HTF bias
        elif htf_bearish and rsi_extreme_short:
            new_signal = -current_size
        
        # === FREQUENCY SAFEGUARD (CRITICAL for trade generation) ===
        # If no trades for 20 bars (~10 days on 12h), allow weaker entry
        if bars_since_last_trade > 20 and new_signal == 0.0 and not in_position:
            if htf_bullish and local_bullish:
                new_signal = BASE_SIZE * 0.8
            elif htf_bearish and local_bearish:
                new_signal = -BASE_SIZE * 0.8
            elif htf_bullish and rsi_oversold:
                new_signal = BASE_SIZE * 0.7
            elif htf_bearish and rsi_overbought:
                new_signal = -BASE_SIZE * 0.7
        
        # === STOPLOSS LOGIC (Rule 6) - 2.5 * ATR ===
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
        
        # === TREND REVERSAL EXIT ===
        trend_reversal = False
        if in_position and position_side != 0:
            # Exit long if 1d trend turns bearish OR local HMA crosses down
            if position_side > 0 and (htf_bearish or hma_cross_short):
                trend_reversal = True
            # Exit short if 1d trend turns bullish OR local HMA crosses up
            if position_side < 0 and (htf_bullish or hma_cross_long):
                trend_reversal = True
        
        # === RSI EXTREME EXIT (take profit) ===
        rsi_exit = False
        if in_position and position_side != 0:
            if position_side > 0 and rsi_14[i] > 75:
                rsi_exit = True
            if position_side < 0 and rsi_14[i] < 25:
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